"""Tests for vcpq.protocol.runner (the model-agnostic cell runner) and its
CLI, vcpq.cli.eval_main (the vcpq-eval console script): scene enumeration,
chunking into input sets, the chunk files of --dump_dir, the predictor
options and the checks of the predicted maps. The scores of vcpq-eval are compared with
vcpq-recompute in tests/test_eval.py.

Run: .venv/bin/python -m pytest tests/test_runner.py -q
"""

from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import pytest

from vcpq.label_systems import load_label_system
from vcpq.protocol.dump import iter_cell, read_chunk_npz
from vcpq.protocol.gt import GT_DIRS
from vcpq.protocol.runner import CellSpec, GroundTruthPredictor, evaluate_cell
from vcpq.cli import eval_main
from vcpq.recompute import RecomputeConfig

from tests._helpers import build_spp100_cell, synthetic_gt_frames, write_gt_frame


def _spec(fixture: dict, n: int, **overrides) -> CellSpec:
    kwargs = dict(
        data_root=fixture['data_root'], dataset=fixture['dataset'],
        label=fixture['label'], frame_list=fixture['frame_list_path'], n=n,
    )
    kwargs.update(overrides)
    return CellSpec(**kwargs)


def _scene_tag(fixture: dict) -> str:
    return f"{fixture['dataset']}__{fixture['scene']}"


def _dump(predictor, spec: CellSpec, raw_dir: str) -> None:
    """Predict, score and write the chunk files of the cell (vcpq-eval --dump_dir)."""
    evaluate_cell(predictor, spec, RecomputeConfig(spec.label), dump_dir=raw_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Chunk files: chunking, GT round trip
# ──────────────────────────────────────────────────────────────────────────────

def test_dump_writes_expected_chunks(tmp_path):
    fixture = build_spp100_cell(tmp_path / 'data', n_frames=12)
    spec = _spec(fixture, n=10)
    raw_dir = str(tmp_path / 'raw')

    _dump(GroundTruthPredictor(spec), spec, raw_dir)

    scene_tag = _scene_tag(fixture)
    c0 = read_chunk_npz(os.path.join(raw_dir, scene_tag, 'c0000.npz'))
    c1 = read_chunk_npz(os.path.join(raw_dir, scene_tag, 'c0001.npz'))
    assert c0.pred_pan.shape[0] == 10
    assert c1.pred_pan.shape[0] == 2
    # GroundTruthPredictor is a perfect predictor: pred == gt exactly.
    np.testing.assert_array_equal(c0.pred_pan, c0.gt_pan)
    np.testing.assert_array_equal(c1.pred_pan, c1.gt_pan)
    assert c0.n == 10 and c1.n == 10
    assert c1.view_offset == 10


def test_dump_matches_make_input_sets_chunking(tmp_path):
    """The dumped stems per chunk must match make_input_sets' own partition —
    not just the counts — since a dumped chunk's view_offset is later looked
    up back against it (vcpq.protocol.input_sets module docstring)."""
    from vcpq.protocol.input_sets import make_input_sets

    fixture = build_spp100_cell(tmp_path / 'data', n_frames=12)
    spec = _spec(fixture, n=10)
    raw_dir = str(tmp_path / 'raw')
    _dump(GroundTruthPredictor(spec), spec, raw_dir)

    expected = make_input_sets(list(fixture['gt_by_stem']), n=10, seed=42)
    scene_tag = _scene_tag(fixture)
    for chunk_idx, (view_offset, stems) in enumerate(expected):
        chunk = read_chunk_npz(os.path.join(raw_dir, scene_tag, f'c{chunk_idx:04d}.npz'))
        assert list(chunk.stems) == stems
        assert chunk.view_offset == view_offset


# ──────────────────────────────────────────────────────────────────────────────
# Scene enumeration (frame list ∩ dataset dir ∩ scenes)
# ──────────────────────────────────────────────────────────────────────────────

def test_scene_enumeration(tmp_path, caplog):
    label = load_label_system('spp100')
    gt_frames = synthetic_gt_frames(4, label)
    data_root = str(tmp_path / 'data')
    dataset = 'scannetpp_spp100'

    frame_list = {}
    for scene in ('sceneA', 'sceneB'):
        scene_dir = os.path.join(data_root, dataset, scene)
        for stem, gt_pan in gt_frames.items():
            write_gt_frame(scene_dir, stem, gt_pan, label)
        frame_list[scene] = list(gt_frames)
    frame_list['sceneC'] = list(gt_frames)  # in the frame list, but never on disk

    frame_list_path = os.path.join(data_root, 'frame_list.json')
    with open(frame_list_path, 'w') as f:
        json.dump(frame_list, f)

    # sceneC is dropped by the dataset-dir intersection, with a warning.
    spec = CellSpec(data_root=data_root, dataset=dataset, label=label,
                    frame_list=frame_list_path, n=10)
    raw_dir = str(tmp_path / 'raw_all')
    with caplog.at_level(logging.WARNING, logger='vcpq.protocol.runner'):
        _dump(GroundTruthPredictor(spec), spec, raw_dir)
    assert {s for s, _ in iter_cell(raw_dir)} == {f'{dataset}__sceneA', f'{dataset}__sceneB'}
    assert '1 of the 3 scenes' in caplog.text and 'sceneC' in caplog.text

    # spec.scenes restricts to just sceneB; nothing requested is missing, no warning.
    spec2 = CellSpec(data_root=data_root, dataset=dataset, label=label,
                     frame_list=frame_list_path, n=10, scenes=('sceneB',))
    raw_dir2 = str(tmp_path / 'raw_scenes')
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='vcpq.protocol.runner'):
        _dump(GroundTruthPredictor(spec2), spec2, raw_dir2)
    assert {s for s, _ in iter_cell(raw_dir2)} == {f'{dataset}__sceneB'}
    assert 'not scored' not in caplog.text


# ──────────────────────────────────────────────────────────────────────────────
# Missing GT handling
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_gt_warns_and_skips(tmp_path, caplog):
    # n_frames == n: single input set (make_input_sets len(stems) <= n branch)
    # -> dropping one stem gives exactly one chunk of 9 views.
    fixture = build_spp100_cell(tmp_path / 'data', n_frames=10)
    dropped_stem = next(iter(fixture['gt_by_stem']))
    inst_dir, sem_dir = GT_DIRS['spp100']
    os.remove(os.path.join(fixture['scene_dir'], inst_dir, f'{dropped_stem}.png'))
    os.remove(os.path.join(fixture['scene_dir'], sem_dir, f'{dropped_stem}.png'))

    spec = _spec(fixture, n=10)
    raw_dir = str(tmp_path / 'raw')
    with caplog.at_level(logging.WARNING, logger='vcpq.protocol.runner'):
        _dump(GroundTruthPredictor(spec), spec, raw_dir)

    written = [p for _, paths in iter_cell(raw_dir) for p in paths]
    assert len(written) == 1
    chunk = read_chunk_npz(written[0])
    assert chunk.pred_pan.shape[0] == 9
    assert dropped_stem not in list(chunk.stems)
    assert dropped_stem in caplog.text


# ──────────────────────────────────────────────────────────────────────────────
# CLI (vcpq.cli.eval_main)
# ──────────────────────────────────────────────────────────────────────────────

def _eval_argv(fixture, out, *extra, predictor='tests.stub_predictor:build'):
    return ['--predictor', predictor, '--data_root', fixture['data_root'],
            '--dataset', fixture['dataset'], '--label_system', 'spp100',
            '--frame_list', fixture['frame_list_path'], '--n', '10',
            '--out', str(out), *extra]


def test_cli_predictor_gt_scores_and_dumps(tmp_path):
    fixture = build_spp100_cell(tmp_path / 'data', n_frames=12)
    raw_dir = str(tmp_path / 'raw')
    out = tmp_path / 'o' / 'result.json'
    eval_main(_eval_argv(fixture, out, '--dump_dir', raw_dir, predictor='gt'))

    result = json.loads(out.read_text())
    assert result['vcpq']['eta100_gamma0.0']['class_avg']['robust_filter_off']['total']['vcpq'] == 1.0
    assert result['pq_scene']['per_scene']['robust_filter_on'] == 1.0
    assert (result['n_scenes'], result['n_chunks']) == (1, 2)
    assert result['raw_dir'] == os.path.abspath(raw_dir)
    scene_tag = _scene_tag(fixture)
    assert os.path.isfile(os.path.join(raw_dir, scene_tag, 'c0000.npz'))
    assert os.path.isfile(os.path.join(raw_dir, scene_tag, 'c0001.npz'))


def test_cli_module_factory_predictor_with_own_args(tmp_path):
    """--predictor module:factory loads build(args); the module's add_args
    options are parsed with the standard ones."""
    fixture = build_spp100_cell(tmp_path / 'data', n_frames=12)
    raw_dir = tmp_path / 'raw'
    eval_main(_eval_argv(fixture, tmp_path / 'r.json', '--sem_id', '2', '--dump_dir', str(raw_dir)))
    chunk = read_chunk_npz(str(raw_dir / _scene_tag(fixture) / 'c0000.npz'))
    assert (chunk.pred_pan == 2000).all()


def test_cli_module_without_add_args_from_current_directory(tmp_path, monkeypatch):
    """add_args is optional, and a module next to the working directory is importable."""
    (tmp_path / 'my_plain_method.py').write_text(
        'import numpy as np\n'
        'from vcpq.protocol.runner import ChunkPrediction\n'
        'class P:\n'
        '    def predict(self, image_paths, stems, gt_size):\n'
        '        return ChunkPrediction([np.full(gt_size, 2000, np.int32) for _ in stems])\n'
        'def make(args):\n'
        '    return P()\n')
    fixture = build_spp100_cell(tmp_path / 'data', n_frames=12)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    eval_main(_eval_argv(fixture, tmp_path / 'r.json', predictor='my_plain_method:make'))
    assert json.loads((tmp_path / 'r.json').read_text())['n_chunks'] == 2
    sys.modules.pop('my_plain_method', None)


def test_cli_help_lists_module_options(capsys):
    with pytest.raises(SystemExit) as exc:
        eval_main(['--predictor', 'tests.stub_predictor:build', '--help'])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for option in ('--sem_id', '--out', '--dump_dir', '--etas', '--gammas', '--robust_filter_threshold',
                   '--include_void_scenes', '--both_robust_filter', '--class_agnostic', '--overwrite'):
        assert option in text, option


@pytest.mark.parametrize('predictor, message', [
    ('panst3r', r"expected 'gt' or 'module:factory'"),
    ('no_such_module_xyz:build', r"module 'no_such_module_xyz' not found"),
    ('tests.stub_predictor:missing', r"has no callable 'missing'"),
    ('tests.stub_predictor:NOT_CALLABLE', r"has no callable 'NOT_CALLABLE'"),
    ('tests.stub_predictor:not_a_predictor', r"returned object, which has no predict"),
])
def test_cli_bad_predictor_is_clear_system_exit(tmp_path, predictor, message):
    fixture = build_spp100_cell(tmp_path / 'data', n_frames=12)
    with pytest.raises(SystemExit, match=message):
        eval_main(_eval_argv(fixture, tmp_path / 'r.json', '--dump_dir', str(tmp_path / 'raw'),
                             predictor=predictor))
    assert not (tmp_path / 'raw').exists() and not (tmp_path / 'r.json').exists()


def test_cli_module_internal_import_failure_is_not_masked(tmp_path, monkeypatch):
    """A module that exists but fails on its own import (a missing third-party
    dependency) is reported with that cause, not as a missing module."""
    (tmp_path / 'needs_dep.py').write_text('import a_dependency_that_is_not_installed_xyz\n')
    fixture = build_spp100_cell(tmp_path / 'data', n_frames=12)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    with pytest.raises(SystemExit, match="importing 'needs_dep' failed: .*a_dependency_that_is_not_installed_xyz"):
        eval_main(_eval_argv(fixture, tmp_path / 'r.json', predictor='needs_dep:build'))


def test_cli_abbreviated_predictor_option_is_rejected(tmp_path):
    fixture = build_spp100_cell(tmp_path / 'data', n_frames=12)
    argv = _eval_argv(fixture, tmp_path / 'r.json')
    argv[0] = '--pred'
    with pytest.raises(SystemExit):
        eval_main(argv)


def test_cli_existing_outputs_need_overwrite(tmp_path, capsys):
    """An existing --out, or chunk files of the cell under --dump_dir, stop
    the command before any prediction unless --overwrite is given."""
    fixture = build_spp100_cell(tmp_path / 'data', n_frames=12)
    out, raw_dir = tmp_path / 'r.json', tmp_path / 'raw'
    eval_main(_eval_argv(fixture, out, '--sem_id', '2', '--dump_dir', str(raw_dir)))
    first = out.read_text()
    capsys.readouterr()

    with pytest.raises(SystemExit):                              # --out exists
        eval_main(_eval_argv(fixture, out, '--sem_id', '1'))
    assert '--overwrite' in capsys.readouterr().err
    with pytest.raises(SystemExit, match='chunk file.*already exist.*--overwrite'):
        eval_main(_eval_argv(fixture, tmp_path / 'other.json', '--sem_id', '1',
                             '--dump_dir', str(raw_dir)))
    assert out.read_text() == first and not (tmp_path / 'other.json').exists()
    assert (read_chunk_npz(str(raw_dir / _scene_tag(fixture) / 'c0000.npz')).pred_pan == 2000).all()

    eval_main(_eval_argv(fixture, out, '--sem_id', '1', '--dump_dir', str(raw_dir), '--overwrite'))
    assert out.read_text() != first
    for name in ('c0000.npz', 'c0001.npz'):
        assert (read_chunk_npz(str(raw_dir / _scene_tag(fixture) / name)).pred_pan == 1000).all()


# ──────────────────────────────────────────────────────────────────────────────
# Prediction validation, empty cell
# ──────────────────────────────────────────────────────────────────────────────

class _ConstPredictor:
    """Predicts one constant id everywhere."""

    def __init__(self, value: int):
        self.value = value

    def predict(self, image_paths, stems, gt_size):
        from vcpq.protocol.runner import ChunkPrediction
        return ChunkPrediction([np.full(gt_size, self.value, np.int32) for _ in stems])


def test_invalid_prediction_rejected(tmp_path):
    fixture = build_spp100_cell(tmp_path / 'data', n_frames=12)
    spec = _spec(fixture, n=10)
    raw_dir = tmp_path / 'raw'
    with pytest.raises(ValueError, match=f"{fixture['scene']} stem .*class not in label system"):
        _dump(_ConstPredictor(150001), spec, str(raw_dir))
    assert not raw_dir.exists()                                 # nothing written


def test_no_scene_raises(tmp_path):
    fixture = build_spp100_cell(tmp_path / 'data', n_frames=12)
    spec = _spec(fixture, n=10, dataset='scannetpp_spp100_wrong')
    raw_dir = tmp_path / 'raw'
    with pytest.raises(ValueError, match='no scene of frame list .*scannetpp_spp100_wrong'):
        _dump(GroundTruthPredictor(spec), spec, str(raw_dir))
    assert not raw_dir.exists()                                 # nothing written
