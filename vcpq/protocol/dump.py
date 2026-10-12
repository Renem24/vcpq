"""The raw chunk dump format: one compressed .npz per (scene, chunk), holding
the pre-robust-filter panoptic maps a metric is computed from.

Keys and dtypes: ``pred_pan``/``gt_pan`` int32 [S, H, W] (S = chunk
size, sem_id * 1000 + inst_id, 0 = void, PRE robust filter), ``stems`` str[S]
and scalar meta (``scene``/``chunk_idx``/``view_offset``/``cs``).
Skip-if-exists by default: a chunk's maps are fixed by the scene, the
input-set size and the predictor, so a prior dump is never silently
clobbered unless ``overwrite=True``.

``Chunk`` is the in-memory form the scorer (``vcpq.recompute.score_scene``)
reads: ``read_chunk_npz`` returns it for a file, ``chunk_from_maps`` builds
the same object from maps without a file (``vcpq-eval``).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from vcpq.label_systems import LabelSystem
from vcpq.panoptic import validate_pred_pan


def chunk_path(raw_dir: str, dataset: str, scene: str, chunk_idx: int) -> str:
    """``{raw_dir}/{dataset}__{scene}/c{chunk_idx:04d}.npz``."""
    return os.path.join(raw_dir, f'{dataset}__{scene}', f'c{chunk_idx:04d}.npz')


def write_chunk_npz(
    path: str,
    pred_pans: list[np.ndarray],
    gt_pans: list[np.ndarray],
    stems: list[str],
    scene: str,
    chunk_idx: int,
    view_offset: int,
    n: int,
    overwrite: bool = False,
    *,
    label: LabelSystem | None = None,
) -> bool:
    """Write one chunk's raw maps. Returns False (no write) if ``path``
    already exists and ``overwrite`` is False.

    With ``label``, every predicted map is checked first
    (``vcpq.panoptic.validate_pred_pan``; ValueError on an invalid id).
    Pass it whenever the maps come from your own code: the scorer does not
    check what it reads.
    """
    if os.path.exists(path) and not overwrite:
        return False
    if label is not None:
        for i, pred in enumerate(pred_pans):
            try:
                validate_pred_pan(pred, label)
            except ValueError as e:
                raise ValueError(f'{path} view {i} ({stems[i]}): {e}') from None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = dict(
        pred_pan=np.stack(pred_pans).astype(np.int32),
        gt_pan=np.stack(gt_pans).astype(np.int32),
        stems=np.array([str(x) for x in stems]),
        scene=np.array(scene),
        chunk_idx=np.int64(chunk_idx),
        view_offset=np.int64(view_offset),
        cs=np.int64(n),
    )
    np.savez_compressed(path, **payload)
    return True


@dataclass
class Chunk:
    """One chunk as the scorer reads it (``read_chunk_npz`` / ``chunk_from_maps``)."""

    pred_pan: np.ndarray
    gt_pan: np.ndarray
    stems: list[str]
    scene: str
    chunk_idx: int
    view_offset: int
    n: int


def read_chunk_npz(path: str) -> Chunk:
    """Read one chunk npz back into a Chunk."""
    with np.load(path) as d:
        return Chunk(
            pred_pan=d['pred_pan'],
            gt_pan=d['gt_pan'],
            stems=list(d['stems']),
            scene=str(d['scene']),
            chunk_idx=int(d['chunk_idx']),
            view_offset=int(d['view_offset']),
            n=int(d['cs']),
        )


def chunk_from_maps(
    pred_pans: list[np.ndarray],
    gt_pans: list[np.ndarray],
    stems: list[str],
    scene: str,
    chunk_idx: int,
    view_offset: int,
    n: int,
) -> Chunk:
    """The ``Chunk`` that ``read_chunk_npz`` returns for the file
    ``write_chunk_npz`` writes from the same arguments (same values and dtypes:
    int32 [S, H, W] maps, numpy-str stems, python scalars), without writing
    it."""
    return Chunk(
        pred_pan=np.stack(pred_pans).astype(np.int32),
        gt_pan=np.stack(gt_pans).astype(np.int32),
        stems=list(np.array([str(x) for x in stems])),
        scene=str(scene),
        chunk_idx=int(chunk_idx),
        view_offset=int(view_offset),
        n=int(n),
    )


def iter_cell(raw_dir: str) -> Iterator[tuple[str, list[str]]]:
    """Yield (scene_tag, sorted c*.npz paths) for every scene dir under
    ``raw_dir``, scene dirs sorted."""
    for scene_tag in sorted(os.listdir(raw_dir)):
        scene_dir = os.path.join(raw_dir, scene_tag)
        if not os.path.isdir(scene_dir):
            continue
        yield scene_tag, sorted(glob.glob(os.path.join(scene_dir, 'c*.npz')))
