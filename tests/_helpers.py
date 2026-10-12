"""Shared test fixture-building helpers (imported by multiple test modules,
not just test_runner.py — keep this dependency-free of any one test file).

The common need: a temp on-disk dataset dir (``color/`` + GT PNGs) that
behaves like a real cell without shipping full-resolution images. Built by
writing synthetic ground-truth label maps (generated here, no dataset
content) as the on-disk PNG pair per frame (inverse of
``vcpq.protocol.gt.load_gt``) plus a tiny dummy color jpg, since nothing here
reads the color image's actual content.
"""

from __future__ import annotations

import json
import os

import numpy as np
import PIL.Image

from vcpq.label_systems import LabelSystem, load_label_system
from vcpq.protocol.gt import GT_DIRS


def synthetic_gt_frames(n_frames: int = 12, label: LabelSystem | None = None,
                        hw: tuple[int, int] = (48, 64), seed: int = 0) -> dict[str, np.ndarray]:
    """``n_frames`` synthetic (stem -> gt_pan [H,W] int32) maps: two stuff
    regions, a few thing rectangles with instance ids 1..k, and a void strip.
    Generated from a seeded RNG; contains no dataset content."""
    label = label or load_label_system('spp100')
    stuff, things = sorted(label.stuff_ids), sorted(label.thing_ids)
    rng = np.random.default_rng(seed)
    h, w = hw
    frames: dict[str, np.ndarray] = {}
    for i in range(n_frames):
        pan = np.zeros((h, w), np.int32)
        pan[: h // 2] = stuff[int(rng.integers(len(stuff)))] * 1000
        pan[h // 2:] = stuff[int(rng.integers(len(stuff)))] * 1000
        for inst in range(1, int(rng.integers(2, 5))):
            y0, x0 = int(rng.integers(0, h - 12)), int(rng.integers(0, w - 16))
            pan[y0: y0 + int(rng.integers(6, 12)), x0: x0 + int(rng.integers(8, 16))] = \
                things[int(rng.integers(len(things)))] * 1000 + inst
        pan[:, : int(rng.integers(1, 4))] = 0
        frames[f'{i:05d}'] = pan
    return frames


def synthetic_prediction(gt_pan: np.ndarray, seed: int = 0) -> np.ndarray:
    """A plausible imperfect prediction of ``gt_pan``: shifted a few pixels
    right, so every class scores strictly between 0 and 1."""
    return np.roll(gt_pan, 3 + seed % 3, axis=1)


def write_gt_frame(scene_dir: str, stem: str, gt_pan: np.ndarray, label: LabelSystem) -> None:
    """Split one gt_pan [H,W] int32 map into the on-disk GT PNG pair plus a
    tiny dummy color jpg, under ``scene_dir``."""
    inst_dir, sem_dir = GT_DIRS[label.name]
    os.makedirs(os.path.join(scene_dir, inst_dir), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, sem_dir), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, 'color'), exist_ok=True)

    sem = (gt_pan // 1000).astype(np.int32)
    thing_ids = np.fromiter(label.thing_ids, dtype=sem.dtype, count=len(label.thing_ids))
    is_thing = np.isin(sem, thing_ids)
    inst = np.where(is_thing, gt_pan % 1000, 0).astype(np.uint16)

    PIL.Image.fromarray(sem.astype(np.uint8)).save(os.path.join(scene_dir, sem_dir, f'{stem}.png'))
    PIL.Image.fromarray(inst).save(os.path.join(scene_dir, inst_dir, f'{stem}.png'))
    PIL.Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
        os.path.join(scene_dir, 'color', f'{stem}.jpg'))


def build_spp100_cell(
    tmp_path,
    n_frames: int = 12,
    dataset: str = 'scannetpp_spp100',
    scene: str = '09c1414f1b',
) -> dict:
    """Build a temp ``{data_root}/{dataset}/{scene}`` spp100 dataset dir from
    synthetic GT (PNG pairs + dummy color jpgs) plus a matching frame-list
    JSON containing exactly that scene's stems.

    Returns a dict with ``data_root``, ``dataset``, ``scene``, ``scene_dir``,
    ``frame_list_path``, ``label`` and ``gt_by_stem`` (the raw gt_pan per
    stem, for round-trip assertions).
    """
    label = load_label_system('spp100')
    gt_by_stem = synthetic_gt_frames(n_frames, label)
    data_root = str(tmp_path)
    scene_dir = os.path.join(data_root, dataset, scene)
    for stem, gt_pan in gt_by_stem.items():
        write_gt_frame(scene_dir, stem, gt_pan, label)

    frame_list_path = os.path.join(data_root, 'frame_list.json')
    with open(frame_list_path, 'w') as f:
        json.dump({scene: list(gt_by_stem)}, f)

    return dict(
        data_root=data_root, dataset=dataset, scene=scene, scene_dir=scene_dir,
        frame_list_path=frame_list_path, label=label, gt_by_stem=gt_by_stem,
    )
