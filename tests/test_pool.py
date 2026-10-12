"""vcpq.pool: pooling the result JSONs of several cells (``vcpq-pool``).

1. Pooling the results of two cells gives the numbers of scoring the chunks
   of both cells together (one raw_dir holding both), with the all-void
   scene left out of the per-scene means or included.
2. Results that differ in label system, include_void_scenes, robust_filter_threshold
   or (eta, gamma) blocks, or lack include_void_scenes or robust_filter_threshold, are
   refused.
3. CLI: the pooled JSON and the printed summary.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tests._helpers import synthetic_gt_frames, synthetic_prediction
from tests.util import assert_leaves_equal
from vcpq.label_systems import load_label_system
from vcpq.pool import main, pool_results
from vcpq.protocol.dump import write_chunk_npz
from vcpq.recompute import RecomputeConfig, recompute_cell

LABEL = load_label_system('spp100')


def _write_scene(raw_dir: Path, tag: str, seed: int, void_gt: bool = False) -> None:
    """Two chunks of three frames: an imperfect prediction of synthetic GT
    (all-void GT with ``void_gt``)."""
    gts = synthetic_gt_frames(6, LABEL, seed=seed)
    stems = list(gts)
    for k in range(2):
        chunk = stems[3 * k: 3 * k + 3]
        preds = [synthetic_prediction(gts[s], i) for i, s in enumerate(chunk)]
        gt = [np.zeros_like(gts[s]) if void_gt else gts[s] for s in chunk]
        write_chunk_npz(str(raw_dir / tag / f'c{k:04d}.npz'), preds, gt, chunk,
                        scene=tag, chunk_idx=k, view_offset=3 * k, n=3)


@pytest.fixture
def two_cells(tmp_path) -> tuple[Path, Path, Path]:
    """Cell A (2 scenes), cell B (2 scenes, one with all-void GT), and one
    raw_dir holding the chunks of both."""
    a, b, both = tmp_path / 'a', tmp_path / 'b', tmp_path / 'both'
    _write_scene(a, 'scannetpp_spp100__s1', seed=1)
    _write_scene(a, 'scannetpp_spp100__s2', seed=2)
    _write_scene(b, 'scannetpp_spp100_iphone__s1', seed=3)
    _write_scene(b, 'scannetpp_spp100_iphone__void', seed=4, void_gt=True)
    for cell in (a, b):
        for scene in cell.iterdir():
            shutil.copytree(scene, both / scene.name)
    return a, b, both


@pytest.mark.parametrize('include_void_scenes, both_robust_filter', ((False, False), (True, True)))
def test_pool_equals_scoring_both_cells_together(two_cells, include_void_scenes, both_robust_filter):
    a, b, both = two_cells
    cfg = RecomputeConfig(LABEL, gammas=(0.0, 1.0), include_void_scenes=include_void_scenes,
                          both_robust_filter=both_robust_filter)
    res_a, _ = recompute_cell(str(a), cfg)
    res_b, _ = recompute_cell(str(b), cfg)
    want, _ = recompute_cell(str(both), cfg)
    got = pool_results([res_a, res_b])

    for key in ('label_system', 'n_scenes', 'n_chunks', 'n_void_scenes', 'include_void_scenes',
                'robust_filter_threshold'):
        assert got[key] == want[key], key
    assert (got['n_scenes'], got['n_void_scenes']) == (4, 1)
    for w, g in [(want['pq_scene'], got['pq_scene'])] + \
            [(want['vcpq'][k], got['vcpq'][k]) for k in want['vcpq']]:
        for part in ('class_avg', 'per_scene', 'per_class'):
            assert_leaves_equal(g[part], w[part])
    # the error decomposition is part of the pooled class averages
    t = got['vcpq']['eta100_gamma0.0']['class_avg']['robust_filter_off']['total']
    assert abs(t['vcpq'] + t['acct_mask'] + t['acct_vc'] + t['acct_thr'] - 1) < 1e-12


def test_pool_refuses_mismatched_results(two_cells):
    a, b, _ = two_cells
    res_a, _ = recompute_cell(str(a), RecomputeConfig(LABEL))
    res_b, _ = recompute_cell(str(b), RecomputeConfig(LABEL))
    with pytest.raises(ValueError, match='label_system'):
        pool_results([res_a, dict(res_b, label_system='scannetv2_s20')])
    with pytest.raises(ValueError, match='include_void_scenes'):
        pool_results([res_a, dict(res_b, include_void_scenes=True)])
    res_rf0, _ = recompute_cell(str(b), RecomputeConfig(LABEL, robust_filter_threshold=0.0))
    with pytest.raises(ValueError, match='robust_filter_threshold'):
        pool_results([res_a, res_rf0])
    for key in ('include_void_scenes', 'robust_filter_threshold'):
        old = {k: v for k, v in res_b.items() if k != key}
        with pytest.raises(ValueError, match=f'no {key} key.*score it again'):
            pool_results([res_a, old])
    res_g1, _ = recompute_cell(str(b), RecomputeConfig(LABEL, gammas=(1.0,)))
    with pytest.raises(ValueError, match='eta, gamma'):
        pool_results([res_a, res_g1])
    res_both, _ = recompute_cell(str(b), RecomputeConfig(LABEL, both_robust_filter=True))
    with pytest.raises(ValueError, match='robust-filtering states'):
        pool_results([res_a, res_both])


def test_cli(two_cells, tmp_path, capsys):
    a, b, _ = two_cells
    paths = []
    for cell in (a, b):
        res, _ = recompute_cell(str(cell), RecomputeConfig(LABEL))
        p = tmp_path / f'{cell.name}.json'
        p.write_text(json.dumps(res))
        paths.append(str(p))
    out = tmp_path / 'o' / 'pooled.json'
    main(paths + ['--out', str(out)])
    printed = capsys.readouterr().out
    assert 'error decomposition' in printed and 'PQ-Scene (paper PQ^scene' in printed
    pooled = json.loads(out.read_text())
    assert pooled['inputs'] == paths
    assert pooled == {'inputs': paths, **pool_results([json.loads(Path(p).read_text()) for p in paths])}
