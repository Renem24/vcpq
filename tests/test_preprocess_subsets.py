"""Tests for the --subset scene/frame selection of the GT preprocessing scripts
(scripts/preprocess_subsets.py) and for the ScanNetv2 label pass with a frame
subset (scripts/preprocess_scannetv2_s20.py:process_labels_s20).

Synthetic data only. The ScanNetv2 script imports imageio, pypng and tqdm at
module level; the label pass does not use the first two, so they are stubbed
here and the test runs in the plain vcpq environment.

Run: .venv/bin/python -m pytest tests/test_preprocess_subsets.py -q
"""

from __future__ import annotations

import importlib.util
import io
import pickle
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from scripts.preprocess_subsets import (
    frame_matches, scannetpp_scenes, scannetv2_scenes, select_frames)
from vcpq.protocol.input_sets import load_frame_list


# --------------------------------------------------------------------------- selection

def test_select_frames_dslr_iphone_scannetv2_stems():
    dslr = ['DSC04432', 'DSC04458', 'DSC04479']
    assert select_frames(dslr, ['04458', '04479']) == (['DSC04458', 'DSC04479'], [])
    iphone = ['frame_000100', 'frame_000104', 'frame_000108']
    assert select_frames(iphone, ['frame_000104']) == (['frame_000104'], [])
    scannet = [str(i) for i in range(12)]
    assert select_frames(scannet, ['7', '4']) == (['4', '7'], [])   # input order kept


def test_select_frames_reports_missing_and_no_loose_match():
    kept, missing = select_frames(['DSC04458'], ['04458', '09999'])
    assert kept == ['DSC04458'] and missing == ['09999']
    # '4458' (leading zero dropped) is not the stem '04458'
    assert not frame_matches('DSC04458', {'4458'})
    assert not frame_matches('frame_000104', {'104'})


def test_scannetpp_scenes_per_subset(tmp_path):
    (tmp_path / 'splits').mkdir()
    (tmp_path / 'splits' / 'nvs_sem_train.txt').write_text('a\nb\n\n')
    (tmp_path / 'splits' / 'nvs_sem_val.txt').write_text('c\nb\n')
    assert scannetpp_scenes(str(tmp_path), 'train', 'dslr') == ['a', 'b']
    assert scannetpp_scenes(str(tmp_path), 'full', 'iphone') == ['a', 'b', 'c']
    for source, name in (('dslr', 'scannetpp_spp100_dslr'),
                         ('iphone', 'scannetpp_spp100_iphone')):
        scenes = scannetpp_scenes(str(tmp_path), 'eval', source)
        assert scenes == list(load_frame_list(name)) and len(scenes) == 50


def test_scannetv2_scenes_per_subset(tmp_path):
    for d in ('scene0002_00', 'scene0001_00', 'other'):
        (tmp_path / d).mkdir()
    (tmp_path / 'scene0003_00.txt').write_text('')           # a file, not a scene dir
    assert scannetv2_scenes(str(tmp_path), 'full') == ['scene0001_00', 'scene0002_00']
    split = tmp_path / 'scannetv2_train.txt'
    split.write_text('scene0002_00\n')
    assert scannetv2_scenes(str(tmp_path), 'train', str(split)) == ['scene0002_00']
    with pytest.raises(ValueError, match='split_file'):
        scannetv2_scenes(str(tmp_path), 'train')
    eval_scenes = scannetv2_scenes(str(tmp_path), 'eval')
    assert len(eval_scenes) == 312 and eval_scenes == list(load_frame_list('scannetv2_s20'))


def test_packaged_lists_have_120_frames_per_scene():
    for name in ('scannetpp_spp100_dslr', 'scannetpp_spp100_iphone', 'scannetv2_s20'):
        assert {len(v) for v in load_frame_list(name).values()} == {120}


# --------------------------------------------------------------------------- ScanNetv2 labels

def _load_scannetv2_script(monkeypatch):
    for name in ('imageio', 'png'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    tqdm_mod = types.ModuleType('tqdm')
    tqdm_mod.tqdm = lambda it, **kw: it
    monkeypatch.setitem(sys.modules, 'tqdm', tqdm_mod)
    path = Path(__file__).resolve().parent.parent / 'scripts' / 'preprocess_scannetv2_s20.py'
    spec = importlib.util.spec_from_file_location('_preprocess_scannetv2_s20', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    PIL.Image.fromarray(arr).save(buf, format='PNG')
    return buf.getvalue()


def _make_scene(root: Path, scene: str) -> Path:
    """Three 4x4 frames. Raw ids: 2 = chair (thing), 3 = floor (stuff).
    Instance 5 is chair in frames 0 and 2 and floor in frame 1, so its class is
    chair only when every frame is counted; instances 3 and 7 are chairs."""
    chair, floor = 2, 3
    labels = [np.full((4, 4), chair, np.uint16),
              np.full((4, 4), floor, np.uint16),
              np.full((4, 4), chair, np.uint16)]
    labels[1][0] = chair
    insts = [np.full((4, 4), 5, np.uint8) for _ in range(3)]
    insts[0][0] = 3
    insts[1][0] = 7
    scene_dir = root / scene
    scene_dir.mkdir()
    with zipfile.ZipFile(scene_dir / f'{scene}_2d-label-filt.zip', 'w') as z:
        for i, a in enumerate(labels):
            z.writestr(f'label-filt/{i}.png', _png(a))
    with zipfile.ZipFile(scene_dir / f'{scene}_2d-instance-filt.zip', 'w') as z:
        for i, a in enumerate(insts):
            z.writestr(f'instance-filt/{i}.png', _png(a))
    return scene_dir


def test_scannetv2_label_subset_matches_full_run(tmp_path, monkeypatch):
    mod = _load_scannetv2_script(monkeypatch)
    scene_dir = _make_scene(tmp_path, 'scene0000_00')
    full, sub = tmp_path / 'full', tmp_path / 'sub'
    ids_full = mod.process_labels_s20(scene_dir, full, num_workers=2)
    ids_sub = mod.process_labels_s20(scene_dir, sub, num_workers=2, frames=['1'])
    assert ids_full == ids_sub == ['0', '1', '2']
    for d in ('s20_semantics', 's20_instance'):
        assert sorted(p.name for p in (sub / d).iterdir()) == ['1.png']
        assert (sub / d / '1.png').read_bytes() == (full / d / '1.png').read_bytes()
    assert (sub / 'segmentation_data.pkl').read_bytes() == \
        (full / 'segmentation_data.pkl').read_bytes()
    # full-scene numbering: 3 -> 1, 5 -> 2, 7 -> 3 (instance 5 stays a chair)
    inst1 = np.asarray(PIL.Image.open(sub / 's20_instance' / '1.png'))
    assert set(np.unique(inst1).tolist()) == {2, 3}
    with open(sub / 'segmentation_data.pkl', 'rb') as f:
        assert pickle.load(f)['s20_instance_to_semantic'] == {0: 0, 1: 2, 2: 2, 3: 2}


def test_scannetv2_label_subset_rejects_unknown_frame(tmp_path, monkeypatch):
    mod = _load_scannetv2_script(monkeypatch)
    scene_dir = _make_scene(tmp_path, 'scene0000_00')
    with pytest.raises(ValueError, match='without labels'):
        mod.process_labels_s20(scene_dir, tmp_path / 'out', num_workers=1, frames=['9'])
