"""Tests for vcpq.protocol.gt (GT loading, frame listing, GT size) and
vcpq.protocol.dump (chunk npz format).

Run: .venv/bin/python -m pytest tests/test_gt_dump.py -q
"""

from __future__ import annotations

import os

import numpy as np
import PIL.Image

from tests._helpers import synthetic_gt_frames
from vcpq.label_systems import load_label_system
from vcpq.protocol.dump import (Chunk, chunk_from_maps, chunk_path, iter_cell, read_chunk_npz,
                                write_chunk_npz)
from vcpq.protocol.gt import GT_DIRS, gt_size, list_frames, load_gt


def _write_gray(path: str, arr: np.ndarray) -> None:
    # No explicit `mode=`: PIL infers 'L' for uint8 / 'I;16' for uint16 from
    # dtype alone, which is the non-deprecated path (mode= to change dtype
    # interpretation is deprecated as of Pillow 13).
    PIL.Image.fromarray(arr).save(path)


# ──────────────────────────────────────────────────────────────────────────────
# gt.py
# ──────────────────────────────────────────────────────────────────────────────

def test_gt_dirs_mapping():
    assert GT_DIRS['spp100'] == ('spp100_instance', 'spp100_semantics')
    assert GT_DIRS['scannetv2_s20'] == ('s20_instance', 's20_semantics')


def test_load_gt_spp100_reconstructs_frame(tmp_path):
    """Split a synthetic spp100 gt_pan back into on-disk PNGs (sem =
    pan // 1000, inst = pan % 1000 — this single formula is correct for BOTH
    things and stuff: for stuff pan is already a multiple of 1000 so inst
    recovers as 0, and for a thing pixel with pan == 0 either the class or its
    instance was genuinely absent on disk, and inst = 0 there triggers the
    same "background instance" void collapse load_gt applies)
    and check load_gt reproduces it exactly, proving the round trip.
    """
    label = load_label_system('spp100')
    stem, gt0 = next(iter(synthetic_gt_frames(1, label).items()))

    inst_dir, sem_dir = GT_DIRS['spp100']
    os.makedirs(tmp_path / inst_dir)
    os.makedirs(tmp_path / sem_dir)
    sem_disk = (gt0 // 1000).astype(np.uint8)
    inst_disk = (gt0 % 1000).astype(np.uint16)
    _write_gray(str(tmp_path / sem_dir / f'{stem}.png'), sem_disk)
    _write_gray(str(tmp_path / inst_dir / f'{stem}.png'), inst_disk)

    reconstructed = load_gt(str(tmp_path), stem, label)
    assert reconstructed.dtype == np.int32
    np.testing.assert_array_equal(reconstructed, gt0)


def test_load_gt_s20_applies_lut_and_zeros_void_instance(tmp_path):
    """200-class on-disk ids remapped through sem_lut to the 20-class
    continuous ids, with instance zeroed wherever the remapped class is void
    (0). Four cases in one row: stuff (wall 1->1), thing with an instance
    (chair 2->5, inst=5), a 200-class id the 20-class taxonomy drops (shelf
    8->0, disk inst=7 must be ignored/zeroed), and a thing pixel with no
    on-disk instance (chair 2->5, inst=0 -> "no instance" collapses to void,
    as load_gt does).
    """
    label = load_label_system('scannetv2_s20')
    assert int(label.sem_lut[1]) == 1   # wall -> stuff 1
    assert int(label.sem_lut[2]) == 5   # chair -> thing 5
    assert int(label.sem_lut[8]) == 0   # shelf -> dropped from s20

    sem_disk = np.array([[1, 2, 8, 2]], dtype=np.uint8)
    inst_disk = np.array([[0, 5, 7, 0]], dtype=np.uint16)

    inst_dir, sem_dir = GT_DIRS['scannetv2_s20']
    os.makedirs(tmp_path / inst_dir)
    os.makedirs(tmp_path / sem_dir)
    _write_gray(str(tmp_path / sem_dir / '0.png'), sem_disk)
    _write_gray(str(tmp_path / inst_dir / '0.png'), inst_disk)

    pan = load_gt(str(tmp_path), '0', label)
    np.testing.assert_array_equal(pan, np.array([[1000, 5005, 0, 0]], dtype=np.int32))


def test_gt_size_reads_header_only(tmp_path):
    inst_dir, sem_dir = GT_DIRS['spp100']
    os.makedirs(tmp_path / inst_dir)
    arr = np.zeros((17, 23), dtype=np.uint16)
    path = str(tmp_path / inst_dir / '0.png')
    _write_gray(path, arr)
    assert gt_size(path) == (17, 23)


def test_list_frames_requires_both_gt_files_numeric_sort(tmp_path):
    label = load_label_system('spp100')
    inst_dir, sem_dir = GT_DIRS['spp100']
    os.makedirs(tmp_path / inst_dir)
    os.makedirs(tmp_path / sem_dir)
    arr = np.zeros((2, 2), dtype=np.uint8)
    for stem in ('00010', '00002', '00001'):
        _write_gray(str(tmp_path / sem_dir / f'{stem}.png'), arr)
        _write_gray(str(tmp_path / inst_dir / f'{stem}.png'), arr.astype(np.uint16))
    # 00001 is missing its instance PNG -> must be dropped.
    os.remove(tmp_path / inst_dir / '00001.png')
    assert list_frames(str(tmp_path), label) == ['00002', '00010']


def test_list_frames_missing_dirs_returns_empty(tmp_path):
    label = load_label_system('spp100')
    assert list_frames(str(tmp_path), label) == []


# ──────────────────────────────────────────────────────────────────────────────
# dump.py
# ──────────────────────────────────────────────────────────────────────────────

def test_chunk_path_format():
    p = chunk_path('/raw', 'scannetv2_s20', 'scene0011_00', 3)
    assert p == '/raw/scannetv2_s20__scene0011_00/c0003.npz'


def _fake_chunk(n=3, h=5, w=7, seed=0):
    rng = np.random.default_rng(seed)
    pred = [rng.integers(0, 5000, size=(h, w), dtype=np.int32) for _ in range(n)]
    gt = [rng.integers(0, 5000, size=(h, w), dtype=np.int32) for _ in range(n)]
    stems = [f'{i:05d}' for i in range(n)]
    return pred, gt, stems


def test_write_read_roundtrip(tmp_path):
    pred, gt, stems = _fake_chunk()
    path = str(tmp_path / 'scannetv2_s20__sceneX' / 'c0000.npz')
    ok = write_chunk_npz(path, pred, gt, stems, scene='scannetv2_s20__sceneX',
                         chunk_idx=0, view_offset=0, n=3)
    assert ok is True
    assert os.path.exists(path)

    chunk = read_chunk_npz(path)
    assert isinstance(chunk, Chunk)
    np.testing.assert_array_equal(chunk.pred_pan, np.stack(pred))
    np.testing.assert_array_equal(chunk.gt_pan, np.stack(gt))
    assert chunk.pred_pan.dtype == np.int32 and chunk.gt_pan.dtype == np.int32
    assert list(chunk.stems) == stems
    assert chunk.scene == 'scannetv2_s20__sceneX'
    assert chunk.chunk_idx == 0
    assert chunk.view_offset == 0
    assert chunk.n == 3


def test_write_skip_if_exists(tmp_path):
    pred, gt, stems = _fake_chunk(seed=1)
    path = str(tmp_path / 'sceneS' / 'c0000.npz')
    assert write_chunk_npz(path, pred, gt, stems, scene='sceneS', chunk_idx=0,
                           view_offset=0, n=3) is True
    mtime1 = os.path.getmtime(path)

    pred2, gt2, stems2 = _fake_chunk(seed=2)
    assert write_chunk_npz(path, pred2, gt2, stems2, scene='sceneS', chunk_idx=0,
                           view_offset=0, n=3) is False
    assert os.path.getmtime(path) == mtime1
    chunk = read_chunk_npz(path)
    np.testing.assert_array_equal(chunk.pred_pan, np.stack(pred))  # unchanged

    assert write_chunk_npz(path, pred2, gt2, stems2, scene='sceneS', chunk_idx=0,
                           view_offset=0, n=3, overwrite=True) is True
    chunk2 = read_chunk_npz(path)
    np.testing.assert_array_equal(chunk2.pred_pan, np.stack(pred2))  # overwritten


def test_iter_cell_sorted_scenes_and_chunks(tmp_path):
    pred, gt, stems = _fake_chunk()
    for scene in ('ds__sceneB', 'ds__sceneA'):
        for idx in (1, 0):
            path = chunk_path(str(tmp_path), 'ds', scene.split('__', 1)[1], idx)
            write_chunk_npz(path, pred, gt, stems, scene=scene, chunk_idx=idx,
                            view_offset=idx * 3, n=3)
    cells = list(iter_cell(str(tmp_path)))
    assert [scene for scene, _ in cells] == ['ds__sceneA', 'ds__sceneB']
    for _, paths in cells:
        assert [os.path.basename(p) for p in paths] == ['c0000.npz', 'c0001.npz']


def test_iter_cell_ignores_non_dir_entries(tmp_path):
    (tmp_path / 'stray_file.txt').write_text('x')
    assert list(iter_cell(str(tmp_path))) == []


def test_write_validates_with_label(tmp_path):
    import pytest
    label = load_label_system('spp100')
    th = min(label.thing_ids)
    gt = [np.zeros((2, 2), np.int32)]
    ok_pred = [np.full((2, 2), th * 1000 + 1, np.int32)]
    bad_pred = [np.full((2, 2), 150001, np.int32)]
    path = str(tmp_path / 's' / 'c0000.npz')
    with pytest.raises(ValueError, match='view 0 \\(00000\\).*class not in label system'):
        write_chunk_npz(path, bad_pred, gt, ['00000'], scene='s', chunk_idx=0,
                        view_offset=0, n=1, label=label)
    assert not os.path.exists(path)
    # without label nothing is checked (existing callers), with a valid map it writes
    assert write_chunk_npz(path, ok_pred, gt, ['00000'], scene='s', chunk_idx=0,
                           view_offset=0, n=1, label=label) is True


def test_chunk_from_maps_equals_read_back(tmp_path):
    """chunk_from_maps (vcpq-eval, no file) builds the same Chunk that
    read_chunk_npz returns for the file write_chunk_npz writes from the same
    arguments: values, dtypes and element types."""
    pred, gt, stems = _fake_chunk(seed=3)
    pred = [p.astype(np.int64) for p in pred]            # converted to int32 on both paths
    for k in range(2):
        path = str(tmp_path / 'ds__s' / f'c{k:04d}.npz')
        args = dict(scene='ds__s', chunk_idx=k, view_offset=3 * k, n=3)
        write_chunk_npz(path, pred, gt, stems, **args)
        want, got = read_chunk_npz(path), chunk_from_maps(pred, gt, stems, **args)
        for name in ('pred_pan', 'gt_pan'):
            a, b = getattr(want, name), getattr(got, name)
            assert a.dtype == b.dtype == np.int32 and a.shape == b.shape
            np.testing.assert_array_equal(a, b)
        assert got.stems == want.stems and [type(x) for x in got.stems] == [type(x) for x in want.stems]
        for name in ('scene', 'chunk_idx', 'view_offset', 'n'):
            assert getattr(got, name) == getattr(want, name)
            assert type(getattr(got, name)) is type(getattr(want, name))
