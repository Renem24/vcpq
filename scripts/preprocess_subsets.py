"""Scene and frame selection for the --subset option of the GT preprocessing scripts.

    eval  : the scenes of the packaged evaluation frame list and, in each scene, only
            the listed frames (120 per scene). The labels of those frames are the
            same as in a full run (see the preprocessing scripts for how).
    train : the official training split, every frame.
    full  : every scene (train + val), every frame.

No third-party dependencies (the tests import it without the rendering stack).
"""

from __future__ import annotations

import os
import os.path as osp

SUBSETS = ('eval', 'train', 'full')

# packaged frame list (vcpq.protocol.input_sets) per image source
EVAL_LISTS = {
    'dslr': 'scannetpp_spp100_dslr',
    'iphone': 'scannetpp_spp100_iphone',
    'scannetv2': 'scannetv2_s20',
}

# ScanNet++ split files under <scannetpp_root>/splits/
SCANNETPP_TRAIN_SPLITS = ('nvs_sem_train',)
SCANNETPP_FULL_SPLITS = ('nvs_sem_train', 'nvs_sem_val')


def eval_frame_list(source: str) -> dict[str, list[str]]:
    """scene -> listed frame stems of the packaged evaluation list for ``source``."""
    from vcpq.protocol.input_sets import load_frame_list
    return load_frame_list(EVAL_LISTS[source])


def read_scene_file(path: str) -> list[str]:
    """One scene id per non-empty line."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _dedup(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def scannetpp_scenes(root: str, subset: str, source: str) -> list[str]:
    """ScanNet++ scenes of ``subset`` (split files are read from <root>/splits/)."""
    if subset == 'eval':
        return list(eval_frame_list(source))
    splits = SCANNETPP_TRAIN_SPLITS if subset == 'train' else SCANNETPP_FULL_SPLITS
    scenes = []
    for split in splits:
        scenes += read_scene_file(osp.join(root, 'splits', f'{split}.txt'))
    return _dedup(scenes)


def scannetv2_scenes(scans_root: str, subset: str, split_file: str | None = None) -> list[str]:
    """ScanNetv2 scenes of ``subset``.

    eval: the packaged list; train: the scenes of ``split_file`` (the official
    ScanNet scannetv2_train.txt); full: every ``scene*`` directory under
    ``scans_root``, sorted.
    """
    if subset == 'eval':
        return list(eval_frame_list('scannetv2'))
    if subset == 'train':
        if split_file is None:
            raise ValueError('--subset train needs --split_file (scannetv2_train.txt)')
        return _dedup(read_scene_file(split_file))
    return sorted(d for d in os.listdir(scans_root)
                  if d.startswith('scene') and osp.isdir(osp.join(scans_root, d)))


def frame_matches(frame_id: str, wanted: set[str]) -> bool:
    """True when ``frame_id`` or its digits are a wanted stem.

    The list stems are 'frame_000104' (iPhone), '04458' (DSLR image DSC04458.JPG)
    and '4' (ScanNetv2 frame index)."""
    if frame_id in wanted:
        return True
    digits = ''.join(ch for ch in frame_id if ch.isdigit())
    return digits in wanted


def select_frames(frame_ids: list[str], wanted: list[str]) -> tuple[list[str], list[str]]:
    """(frame_ids that match a wanted stem, in input order; wanted stems that no
    frame matched)."""
    wanted_set = set(wanted)
    kept = [fid for fid in frame_ids if frame_matches(fid, wanted_set)]
    found = set()
    for fid in kept:
        found.add(fid)
        found.add(''.join(ch for ch in fid if ch.isdigit()))
    missing = [s for s in wanted if s not in found]
    return kept, missing
