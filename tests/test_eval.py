"""vcpq-eval (vcpq.cli.eval_main, vcpq.protocol.runner.evaluate_cell): the
predictor is scored in memory, and the numbers are those of vcpq-recompute
on a dump of the same predictions.

1. ``vcpq-eval --dump_dir D --out A.json`` and ``vcpq-recompute D`` (serial
   and ``--workers 2``) give identical JSON leaves (floats exact); the chunk
   records of ``evaluate_cell`` equal those of ``recompute_chunks.json.gz``.
2. Without ``--dump_dir`` nothing but the JSON is written, with the same
   scores and ``raw_dir`` null.
3. The predictor is called once per chunk.

Synthetic cell: four scenes of 12 frames, one with all-void ground truth,
N = 5 (input sets of 5, 5 and 2 views); ``tests.stub_predictor:build_noisy``.
"""
from __future__ import annotations

import glob
import gzip
import json
import os

import pytest

from tests import stub_predictor
from tests._helpers import synthetic_gt_frames, write_gt_frame
from tests.util import assert_leaves_equal
from vcpq.cli import eval_main
from vcpq.label_systems import load_label_system
from vcpq.protocol.input_sets import make_input_sets
from vcpq.protocol.runner import CellSpec, evaluate_cell
from vcpq.recompute import RecomputeConfig
from vcpq.recompute import main as recompute_main

DATASET = 'scannetpp_spp100'
SCENES = ('aaa', 'bbb', 'ccc', 'ddd')
VOID_SCENE = 'ccc'
SCORING = ['--gammas', '0.0', '1.0', '--etas', '50', '100']


@pytest.fixture
def cell(tmp_path) -> dict:
    label = load_label_system('spp100')
    data_root = str(tmp_path / 'data')
    frame_list = {}
    for k, scene in enumerate(SCENES):
        gts = synthetic_gt_frames(12, label, seed=k)
        for stem, gt in gts.items():
            write_gt_frame(os.path.join(data_root, DATASET, scene), stem,
                           gt * 0 if scene == VOID_SCENE else gt, label)
        frame_list[scene] = list(gts)
    path = os.path.join(data_root, 'frame_list.json')
    with open(path, 'w') as f:
        json.dump(frame_list, f)
    return dict(data_root=data_root, frame_list=path, frame_list_dict=frame_list)


def _eval_argv(cell: dict, out, *extra) -> list[str]:
    return ['--predictor', 'tests.stub_predictor:build_noisy', '--data_root', cell['data_root'],
            '--dataset', DATASET, '--label_system', 'spp100', '--frame_list', cell['frame_list'],
            '--n', '5', *SCORING, '--out', str(out), *extra]


def _load(path) -> dict:
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as f:
        return json.load(f)


def _without(d: dict, *keys: str) -> dict:
    return {k: v for k, v in d.items() if k not in keys}


@pytest.mark.parametrize('options', ([], ['--include_void_scenes', '--robust_filter_threshold', '0']))
def test_eval_equals_recompute_of_its_dump(cell, tmp_path, options):
    dump = tmp_path / 'dump'
    eval_main(_eval_argv(cell, tmp_path / 'eval.json', '--dump_dir', str(dump), *options))
    got = _load(tmp_path / 'eval.json')
    assert (got['n_scenes'], got['n_chunks'], got['n_void_scenes']) == (4, 12, 1)
    assert got['raw_dir'] == os.path.abspath(dump)
    assert list(got['vcpq']) == ['eta50_gamma0.0', 'eta50_gamma1.0', 'eta100_gamma0.0', 'eta100_gamma1.0']
    vc = got['vcpq']['eta100_gamma0.0']['class_avg']['robust_filter_off']['total']['vcpq']
    assert 0 < vc < 1                                             # an imperfect prediction

    for workers in ('1', '2'):
        out = tmp_path / f'recompute_w{workers}' / 'recompute.json'
        recompute_main([str(dump), '--label_system', 'spp100', *SCORING, *options,
                        '--workers', workers, '--out', str(out)])
        # everything, raw_dir included, except the chunks-file path of --workers 2
        assert_leaves_equal(_without(got, 'chunks_file'), _without(_load(out), 'chunks_file'), rel=0)

    # per-chunk records: evaluate_cell (Python API) vs recompute_chunks.json.gz
    cfg = RecomputeConfig(load_label_system('spp100'), etas=(50, 100), gammas=(0.0, 1.0),
                          robust_filter_threshold=0.0 if options else 0.005,
                          include_void_scenes=bool(options))
    spec = CellSpec(data_root=cell['data_root'], dataset=DATASET, label=cfg.label,
                    frame_list=cell['frame_list'], n=5)
    result, records = evaluate_cell(stub_predictor.NoisyPredictor(cfg.label), spec, cfg)
    assert_leaves_equal(_without(got, 'raw_dir'), _without(result, 'raw_dir'), rel=0)
    chunks = _load(tmp_path / 'recompute_w2' / 'recompute_chunks.json.gz')['chunks']
    assert_leaves_equal(chunks, json.loads(json.dumps(records)), rel=0)


def test_eval_without_dump_writes_only_the_json(cell, tmp_path):
    eval_main(_eval_argv(cell, tmp_path / 'with' / 'eval.json', '--dump_dir', str(tmp_path / 'dump')))
    eval_main(_eval_argv(cell, tmp_path / 'without' / 'eval.json'))
    assert os.listdir(tmp_path / 'without') == ['eval.json']
    assert sorted(glob.glob(str(tmp_path / '**' / '*.npz'), recursive=True)) == \
        sorted(glob.glob(str(tmp_path / 'dump' / '*' / '*.npz')))
    with_dump, without = _load(tmp_path / 'with' / 'eval.json'), _load(tmp_path / 'without' / 'eval.json')
    assert without['raw_dir'] is None
    assert_leaves_equal(_without(with_dump, 'raw_dir'), _without(without, 'raw_dir'), rel=0)


@pytest.mark.parametrize('dump', (False, True))
def test_eval_calls_predictor_once_per_chunk(cell, tmp_path, dump):
    extra = ['--dump_dir', str(tmp_path / 'dump')] if dump else []
    stub_predictor.built.clear()
    eval_main(_eval_argv(cell, tmp_path / 'eval.json', *extra))
    [predictor] = stub_predictor.built
    expected = [(scene, tuple(stems)) for scene in SCENES
                for _, stems in make_input_sets(cell['frame_list_dict'][scene], 5, seed=42)]
    assert predictor.predicted == expected                        # each chunk once, in order
    assert _load(tmp_path / 'eval.json')['n_chunks'] == len(expected) == 12
    if dump:
        assert len(glob.glob(str(tmp_path / 'dump' / '*' / 'c*.npz'))) == len(expected)
