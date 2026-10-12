"""vcpq.recompute: the offline scorer (raw chunk dump -> recompute.json).

1. A perfect (GT) predictor scores VC-PQ = PQ^scene = 1. The default run
   computes VC-PQ with robust filtering off and PQ^scene with it on
   (class-wise matching) and equals those variants of a run with
   ``both_robust_filter`` and ``class_agnostic``.
2. All-void-GT scenes: left out of the per-scene means by default and
   counted in ``n_void_scenes``; with ``include_void_scenes=True`` the scene
   enters with its value (0). Pooled (class-avg) numbers are the same either
   way.
3. CLI: output files, ``chunks_file``, chunk header, help text.
4. tests/util.py (leaf comparison).
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests._helpers import build_spp100_cell, synthetic_gt_frames, synthetic_prediction
from tests.util import assert_leaves_equal
from vcpq.label_systems import load_label_system
from vcpq.protocol.dump import write_chunk_npz
from vcpq.protocol.runner import CellSpec, GroundTruthPredictor, evaluate_cell
from vcpq.pq_scene_accumulator import PQSceneAccumulator
from vcpq.recompute import RecomputeConfig, main, recompute_cell
from vcpq.vcpq_accumulator import VCPQAccumulator

FAMILIES = ('vcpq', 'pq_scene')
G0 = 'eta100_gamma0.0'


# ── 1. perfect predictor ─────────────────────────────────────────────────────

@pytest.fixture
def gt_cell_raw_dir(tmp_path) -> str:
    """A 12-frame spp100 cell dumped by GroundTruthPredictor (pred == GT), N=10."""
    fx = build_spp100_cell(tmp_path / 'data', n_frames=12)
    spec = CellSpec(data_root=fx['data_root'], dataset=fx['dataset'], label=fx['label'],
                    frame_list=fx['frame_list_path'], n=10)
    raw = str(tmp_path / 'raw')
    evaluate_cell(GroundTruthPredictor(spec), spec, RecomputeConfig(fx['label']), dump_dir=raw)
    return raw


def test_gt_predictor_is_perfect(gt_cell_raw_dir):
    label = load_label_system('spp100')
    res, _ = recompute_cell(gt_cell_raw_dir, RecomputeConfig(label))
    vc = res['vcpq'][G0]
    # default: VC-PQ with robust filtering off, PQ^scene with it on, class-wise only
    assert list(vc['class_avg']) == ['robust_filter_off'] and list(res['pq_scene']['per_scene']) == ['robust_filter_on']
    assert 'class_agnostic' not in vc and 'class_agnostic' not in res['pq_scene']
    assert vc['class_avg']['robust_filter_off']['total']['vcpq'] == 1.0 and vc['per_scene'] == {'robust_filter_off': 1.0}
    assert res['pq_scene']['per_scene'] == {'robust_filter_on': 1.0}
    full, _ = recompute_cell(gt_cell_raw_dir, RecomputeConfig(label, both_robust_filter=True, class_agnostic=True))
    vc = full['vcpq'][G0]
    assert vc['class_avg']['robust_filter_on']['total']['vcpq'] == 1.0
    assert vc['per_scene'] == {'robust_filter_off': 1.0, 'robust_filter_on': 1.0}
    assert vc['class_agnostic']['robust_filter_off']['vcpq'] == vc['class_agnostic']['things_robust_filter_on']['vcpq'] == 1.0
    assert full['pq_scene']['class_avg']['robust_filter_off']['total']['pq'] == 1.0
    assert full['pq_scene']['per_scene'] == {'robust_filter_off': 1.0, 'robust_filter_on': 1.0}
    assert list(res) == ['raw_dir', 'label_system', 'n_scenes', 'n_chunks', 'n_void_scenes',
                         'include_void_scenes', 'robust_filter_threshold', 'pq_scene', 'vcpq']


# ── 2. all-void-GT scenes ────────────────────────────────────────────────────

def _write_scene(raw_dir: Path, tag: str, void_gt: bool, n_frames: int = 3) -> None:
    """One synthetic chunk of ``n_frames`` frames (imperfect prediction of a
    synthetic GT); ``void_gt`` replaces its GT by all-void (the prediction
    stays a real one)."""
    gts = synthetic_gt_frames(n_frames)
    stems = list(gts)
    preds = [synthetic_prediction(gts[s], i) for i, s in enumerate(stems)]
    gt = [np.zeros_like(gts[s]) for s in stems] if void_gt else [gts[s] for s in stems]
    write_chunk_npz(str(raw_dir / tag / 'c0000.npz'), preds, gt,
                    stems, scene=tag, chunk_idx=0, view_offset=0, n=10)


def _write_void_scene(raw_dir: Path, tag: str = 'scannetpp_spp100__void') -> None:
    _write_scene(raw_dir, tag, void_gt=True)


def _per_scene(res: dict) -> dict:
    return {'vcpq': res['vcpq'][G0]['per_scene'], 'pq_scene': res['pq_scene']['per_scene']}


def test_void_scene_excluded_by_default(tmp_path):
    raw = tmp_path / 'raw'
    _write_void_scene(raw)
    res, _ = recompute_cell(str(raw), RecomputeConfig(load_label_system('spp100'), both_robust_filter=True))
    assert res['n_void_scenes'] == 1 and res['n_scenes'] == 1
    # no scene left for the per-scene mean
    assert _per_scene(res) == {f: {'robust_filter_off': 0.0, 'robust_filter_on': 0.0} for f in FAMILIES}
    inc, _ = recompute_cell(str(raw), RecomputeConfig(load_label_system('spp100'), both_robust_filter=True,
                                                      include_void_scenes=True))
    assert inc['n_void_scenes'] == 1
    # include_void_scenes: the void scene enters with its own value, which is 0
    assert _per_scene(inc) == {f: {'robust_filter_off': 0.0, 'robust_filter_on': 0.0} for f in FAMILIES}


@pytest.mark.parametrize('workers', (1, 2))
def test_void_scene_next_to_real_scene(tmp_path, workers):
    label = load_label_system('spp100')
    real_only = tmp_path / 'real_only'
    _write_scene(real_only, 'scannetpp_spp100__real', void_gt=False)
    one = _per_scene(recompute_cell(str(real_only), RecomputeConfig(label))[0])
    raw = tmp_path / 'raw'
    _write_scene(raw, 'scannetpp_spp100__real', void_gt=False)
    _write_void_scene(raw)
    dflt, _ = recompute_cell(str(raw), RecomputeConfig(label, workers=workers))
    inc, _ = recompute_cell(str(raw), RecomputeConfig(label, workers=workers, include_void_scenes=True))
    for r in (dflt, inc):
        assert (r['n_scenes'], r['n_chunks'], r['n_void_scenes']) == (2, 2, 1)
    assert all(v > 0 for f in FAMILIES for v in one[f].values())   # the real scene scores
    assert _per_scene(dflt) == one                                  # void scene left out
    assert _per_scene(inc) == {f: {rf: float(np.mean([v, 0.0])) for rf, v in one[f].items()}
                               for f in FAMILIES}                   # void scene enters as 0
    for fam in FAMILIES:                                            # pooled: unaffected
        a = dflt[fam][G0] if fam == 'vcpq' else dflt[fam]
        b = inc[fam][G0] if fam == 'vcpq' else inc[fam]
        assert a['class_avg'] == b['class_avg'] and a['per_class'] == b['per_class'], fam



@pytest.mark.parametrize('workers', (1, 2))
def test_default_equals_those_variants_of_the_full_run(tmp_path, workers):
    """The default run gives exactly (bit for bit) the numbers of the same
    variants in a run with both_robust_filter and class_agnostic, per-chunk records too."""
    label = load_label_system('spp100')
    raw = tmp_path / 'raw'
    _write_scene(raw, 'scannetpp_spp100__a', void_gt=False)
    _write_scene(raw, 'scannetpp_spp100__b', void_gt=False, n_frames=5)
    _write_void_scene(raw)
    opts = dict(gammas=(0.0, 1.0), workers=workers)
    dflt, rec_d = recompute_cell(str(raw), RecomputeConfig(label, **opts))
    full, rec_f = recompute_cell(str(raw), RecomputeConfig(label, both_robust_filter=True, class_agnostic=True,
                                                           **opts))
    pairs = [('pq_scene', dflt['pq_scene'], full['pq_scene'])] + \
        [('vcpq', dflt['vcpq'][k], full['vcpq'][k]) for k in full['vcpq']]
    for fam, d, f in pairs:
        state = 'robust_filter_on' if fam == 'pq_scene' else 'robust_filter_off'
        assert set(d) == {'class_avg', 'per_scene', 'per_class'} | ({'accounting'} if fam == 'vcpq' else set())
        for part in d:
            assert list(d[part]) == [state], (fam, part)
            assert_leaves_equal(d[part], {state: f[part][state]}, rel=0)
    if workers > 1:
        assert len(rec_d) == len(rec_f) == 3
        for cd, cf in zip(rec_d, rec_f):
            assert cd['pq_scene'] == {'robust_filter_on': cf['pq_scene']['robust_filter_on']}
            assert cd['vcpq'] == {k: {'robust_filter_off': v['robust_filter_off']} for k, v in cf['vcpq'].items()}


def test_accumulators_refuse_a_variant_not_computed():
    label = load_label_system('spp100')
    with pytest.raises(ValueError, match='robust_filter_on variant was not computed'):
        VCPQAccumulator(label, robust_filter='off').compute_metrics(rf=True)
    with pytest.raises(ValueError, match='robust_filter_off variant was not computed'):
        PQSceneAccumulator(label, robust_filter='on').raw_stats()
    with pytest.raises(ValueError, match="robust_filter must be 'off', 'on' or 'both'"):
        VCPQAccumulator(label, robust_filter='none')

# ── 3. CLI ───────────────────────────────────────────────────────────────────

def test_cli_main(gt_cell_raw_dir, tmp_path, capsys):
    out = tmp_path / 'o' / 'recompute.json'
    main([gt_cell_raw_dir, '--label_system', 'spp100', '--workers', '2', '--gammas', '0.0', '1.0',
          '--out', str(out)])
    printed = capsys.readouterr().out
    assert 'scored: 1 scenes, 2 chunks' in printed
    assert 'VC-PQ eta100_gamma0.0 (paper VC-PQ, class-avg, robust filtering off)' in printed
    assert 'VC-PQ eta100_gamma1.0 (class-avg, robust filtering off)' in printed       # paper's VC-PQ is gamma 0
    assert 'PQ-Scene (paper PQ^scene, per-scene mean, robust filtering on): 100.0' in printed
    assert 'PQ-Scene secondary, pooled class-avg' in printed
    got = json.loads(out.read_text())
    assert list(got['vcpq']) == [G0, 'eta100_gamma1.0']
    assert got['vcpq'][G0]['class_avg']['robust_filter_off']['total']['vcpq'] == 1.0
    assert got['chunks_file'] == 'recompute_chunks.json.gz'
    with gzip.open(out.parent / got['chunks_file'], 'rt') as f:
        payload = json.load(f)
    assert payload['robust_filter_threshold'] == 0.005 and payload['gammas'] == [0.0, 1.0]
    assert sum(c['n_frames'] for c in payload['chunks']) == 12

    out2 = tmp_path / 'p' / 'recompute.json'
    main([gt_cell_raw_dir, '--label_system', 'spp100', '--workers', '2', '--no_chunks',
          '--out', str(out2)])
    assert 'chunks_file' not in json.loads(out2.read_text())
    assert os.listdir(out2.parent) == ['recompute.json']

    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        main(['--help'])
    assert e.value.code == 0
    text = ' '.join(capsys.readouterr().out.split())
    assert '--include_void_scenes' in text
    assert 'VC-PQ only' in text and 'default: 0.005' in text


# ── 4. tests/util.py ─────────────────────────────────────────────────────────

def test_util_leaf_compare():
    a = {'x': 1, 'y': {'z': 0.1, 'w': [1, 2.5]}}
    assert_leaves_equal(a, {'x': 1, 'y': {'z': 0.1 * (1 + 1e-12), 'w': [1, 2.5]}})
    with pytest.raises(AssertionError, match='/y/z'):
        assert_leaves_equal(a, {'x': 1, 'y': {'z': 0.1 * (1 + 1e-6), 'w': [1, 2.5]}})
    with pytest.raises(AssertionError, match='/x'):
        assert_leaves_equal(a, {'x': 2, 'y': a['y']})              # ints exact
    with pytest.raises(AssertionError, match='2 differing leaves.*missing'):
        assert_leaves_equal(a, {'x': 1, 'y': {'z': 0.1}})         # /y/w has 2 leaves
    with pytest.raises(AssertionError, match='missing in old'):
        assert_leaves_equal(a, dict(a, extra=0))
    assert_leaves_equal(a, dict(a, extra=0), subset=True)

