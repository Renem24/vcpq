"""Ground-truth panoptic map loading for the two label systems.

``list_frames`` gives the frames a scene is evaluated on: only stems with GT
on both sides are usable.

Both label systems store GT as a pair of PNGs per frame: ``{stem}.png`` under
an instance subdir (uint16) and a semantic subdir (uint8). spp100 stores its
own continuous class ids directly (no remap). scannetv2_s20 stores the
200-class ids; ``LabelSystem.sem_lut`` remaps them to the 20-class continuous
ids, and the instance map is zeroed wherever the remapped class is void — a
200-class id the 20-class taxonomy drops (e.g. "shelf") is not a s20 class at
all, so any instance under it must not survive the remap either.
"""

from __future__ import annotations

import os

import numpy as np
import PIL.Image

from vcpq.label_systems import LabelSystem
from vcpq.panoptic import encode_panoptic

# {label system name: (instance subdir, semantic subdir)}
GT_DIRS: dict[str, tuple[str, str]] = {
    'spp100': ('spp100_instance', 'spp100_semantics'),
    'scannetv2_s20': ('s20_instance', 's20_semantics'),
}


def gt_size(inst_png: str) -> tuple[int, int]:
    """(H, W) of a GT frame, read from the PNG header only (no decode)."""
    with PIL.Image.open(inst_png) as im:
        w, h = im.size
    return h, w


def load_gt(scene_dir: str, stem: str, label: LabelSystem) -> np.ndarray:
    """Load one frame's GT panoptic map.

    Returns:
        [H, W] int32 panoptic map (sem_id * 1000 + inst_id, 0 = void), in
        ``label``'s continuous id space.
    """
    inst_dir, sem_dir = GT_DIRS[label.name]
    inst_path = os.path.join(scene_dir, inst_dir, stem + '.png')
    sem_path = os.path.join(scene_dir, sem_dir, stem + '.png')
    inst = np.array(PIL.Image.open(inst_path)).astype(np.int32)
    sem = np.array(PIL.Image.open(sem_path)).astype(np.int32)

    if label.sem_lut is not None:
        sem = label.sem_lut[sem]
        inst = np.where(sem == 0, 0, inst)

    return encode_panoptic(sem, inst, label.thing_ids)


def _sort_key(stem: str):
    return int(stem) if stem.isdigit() else stem


def list_frames(scene_dir: str, label: LabelSystem) -> list[str]:
    """Stems with BOTH GT PNGs present, numeric-sorted when the stem is a digit
    string (else lexicographic)."""
    inst_dir, sem_dir = GT_DIRS[label.name]
    inst_path = os.path.join(scene_dir, inst_dir)
    sem_path = os.path.join(scene_dir, sem_dir)
    if not os.path.isdir(inst_path) or not os.path.isdir(sem_path):
        return []
    inst_stems = {os.path.splitext(f)[0] for f in os.listdir(inst_path)
                  if f.lower().endswith('.png')}
    sem_stems = {os.path.splitext(f)[0] for f in os.listdir(sem_path)
                 if f.lower().endswith('.png')}
    return sorted(inst_stems & sem_stems, key=_sort_key)
