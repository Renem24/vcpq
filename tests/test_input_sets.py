"""Tests for the input-set protocol (vcpq.protocol.input_sets).

The chunking rule (make_input_sets): shuffle a scene's frame indices with
random.Random(seed), slice into fixed-size chunks, sort each chunk
numerically.

Run: .venv/bin/python -m pytest tests/test_input_sets.py -q
"""

from __future__ import annotations

import importlib.resources
import json

from vcpq.protocol.input_sets import load_frame_list, make_input_sets


def test_counts_per_scene():
    stems = [f'{i:05d}' for i in range(120)]
    assert [len(make_input_sets(stems, n)) for n in (5, 10, 20, 40)] == [24, 12, 6, 3]


def test_short_scene_single_set():
    assert make_input_sets(['1', '2', '3'], 10) == [(0, ['1', '2', '3'])]


def test_exact_short_boundary():
    # len(stems) == n is still the single-set branch (chunking starts only when
    # len(stems) > n).
    stems = [f'{i:05d}' for i in range(10)]
    assert make_input_sets(stems, 10) == [(0, stems)]


def test_load_frame_list_packaged_names():
    for name, n_scenes in (
        ('scannetpp_spp100_dslr', 50),
        ('scannetpp_spp100_iphone', 50),
        ('scannetv2_s20', 312),
    ):
        fl = load_frame_list(name)
        assert len(fl) == n_scenes
        assert all(len(v) == 120 for v in fl.values())


def test_load_frame_list_path(tmp_path):
    p = tmp_path / 'custom.json'
    p.write_text(json.dumps({'sceneA': ['1', '2', '3']}))
    assert load_frame_list(str(p)) == {'sceneA': ['1', '2', '3']}


def test_tail_set_logs_warning(caplog):
    import logging
    stems = [f'{i:05d}' for i in range(12)]
    with caplog.at_level(logging.WARNING, logger='vcpq.protocol.input_sets'):
        sets = make_input_sets(stems, 10)
    assert [len(s) for _, s in sets] == [10, 2]                 # behaviour unchanged
    assert 'last input set has 2 frames' in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='vcpq.protocol.input_sets'):
        make_input_sets([f'{i:05d}' for i in range(120)], 10)
    assert caplog.text == ''


def test_packaged_input_set_files_match_make_input_sets():
    """The written-out input sets (input_sets_<list>_N<n>.json) are
    make_input_sets of the packaged frame lists, in list order; the iPhone
    files leave out d755b3d9d8 (all-void ground truth)."""
    lists = importlib.resources.files('vcpq.protocol') / 'lists'
    for name in ('scannetpp_spp100_dslr', 'scannetpp_spp100_iphone', 'scannetv2_s20'):
        frame_list = load_frame_list(name)
        for n in (5, 10, 20, 40):
            rows = json.loads((lists / f'input_sets_{name}_N{n}.json').read_text())
            want = [{'scene': scene, 'index': i, 'view_offset': offset, 'stems': stems}
                    for scene, scene_stems in frame_list.items()
                    if not (name == 'scannetpp_spp100_iphone' and scene == 'd755b3d9d8')
                    for i, (offset, stems) in enumerate(make_input_sets(scene_stems, n))]
            assert rows == want, (name, n)
