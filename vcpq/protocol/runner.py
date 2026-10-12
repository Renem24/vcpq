"""The model-agnostic cell runner: runs any ``ChunkPredictor`` over the
paper's input sets of a cell.

``iter_scene_chunks`` enumerates one scene's input sets, calls the predictor
once per input set, validates the maps and pairs them with the ground truth
as a ``vcpq.protocol.dump.Chunk``. ``evaluate_cell`` (``vcpq-eval``) scores
the chunks in memory, one scene at a time, with the scorer of
``vcpq.recompute`` and returns the ``recompute.json`` result; with
``dump_dir`` it also writes every chunk
(``vcpq.protocol.dump.write_chunk_npz``), for ``vcpq-recompute`` to score
again later.
"""

from __future__ import annotations

import glob
import itertools
import logging
import os
from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np

from vcpq.label_systems import LabelSystem
from vcpq.panoptic import validate_pred_pan
from vcpq.protocol.dump import Chunk, chunk_from_maps, chunk_path, write_chunk_npz
from vcpq.protocol.gt import GT_DIRS, gt_size, list_frames, load_gt
from vcpq.protocol.input_sets import load_frame_list, make_input_sets
from vcpq.recompute import RecomputeConfig, SceneResult, aggregate_scenes, cell_result, score_scene

logger = logging.getLogger('vcpq.protocol.runner')


@dataclass
class ChunkPrediction:
    """One chunk's predicted panoptic maps, in the same frame order as the
    ``stems`` a ``ChunkPredictor.predict`` call was given."""

    pred_pans: list[np.ndarray]              # F x int32 [H_gt, W_gt], sem*1000+inst


class ChunkPredictor(Protocol):
    def predict(self, image_paths: list[str], stems: list[str],
                gt_size: tuple[int, int]) -> ChunkPrediction: ...


@dataclass
class CellSpec:
    """One (dataset, label system, frame list, chunk size) cell."""

    data_root: str
    dataset: str
    label: LabelSystem
    frame_list: str
    n: int
    seed: int = 42
    scenes: tuple[str, ...] | None = None


def resolve_image_path(scene_dir: str, stem: str) -> str:
    """``{scene_dir}/color/{stem}.<ext>`` — extension resolved by globbing,
    since spp100 and s20 both use ``color/`` but store jpgs (today)."""
    matches = sorted(glob.glob(os.path.join(scene_dir, 'color', stem + '.*')))
    if not matches:
        raise FileNotFoundError(f'no color image for stem {stem!r} under {scene_dir}/color')
    return matches[0]


def cell_scenes(spec: CellSpec) -> list[str]:
    """The scenes of the cell: sorted(frame list) ∩ dataset dir ∩ spec.scenes."""
    frame_list = load_frame_list(spec.frame_list)
    dataset_dir = os.path.join(spec.data_root, spec.dataset)
    on_disk: set[str] = set()
    if os.path.isdir(dataset_dir):
        on_disk = {d for d in os.listdir(dataset_dir)
                   if not d.startswith('.') and os.path.isdir(os.path.join(dataset_dir, d))}

    scenes = [s for s in sorted(frame_list) if s in on_disk]
    if spec.scenes is not None:
        wanted = set(spec.scenes)
        scenes = [s for s in scenes if s in wanted]
    return scenes


def usable_stems(spec: CellSpec, scene: str) -> list[str]:
    """The scene's frame-list stems that have GT on both sides, in frame-list
    order — the frames ``iter_scene_chunks`` chunks. Each dropped stem is
    logged as a warning.
    """
    wanted_stems = load_frame_list(spec.frame_list)[scene]
    available = set(list_frames(os.path.join(spec.data_root, spec.dataset, scene), spec.label))
    for stem in wanted_stems:
        if stem not in available:
            logger.warning('%s/%s: GT missing for stem %s, skipping', spec.dataset, scene, stem)
    return [s for s in wanted_stems if s in available]


def iter_scene_chunks(
    predictor: ChunkPredictor,
    spec: CellSpec,
    scene: str,
) -> Iterator[Chunk]:
    """Predict every chunk of one scene and yield it with its ground truth,
    in chunk order, one chunk at a time.

    Frame-list stems without GT on both sides are dropped, with a warning,
    before chunking (``usable_stems``).

    Every predicted map is checked (dtype, shape and
    ``vcpq.panoptic.validate_pred_pan``) before the chunk is yielded;
    ValueError names the scene, the stem and the offending ids.

    Yields:
        ``Chunk`` (scene tag ``<dataset>__<scene>``), the same object
        ``read_chunk_npz`` returns for the chunk's npz file.
    """
    stems = usable_stems(spec, scene)
    scene_dir = os.path.join(spec.data_root, spec.dataset, scene)
    inst_dir, _sem_dir = GT_DIRS[spec.label.name]
    if not stems:
        return

    for chunk_idx, (view_offset, chunk_stems) in enumerate(
            make_input_sets(stems, spec.n, seed=spec.seed)):
        image_paths = [resolve_image_path(scene_dir, s) for s in chunk_stems]
        h, w = gt_size(os.path.join(scene_dir, inst_dir, chunk_stems[0] + '.png'))

        prediction = predictor.predict(image_paths, chunk_stems, (h, w))
        if len(prediction.pred_pans) != len(chunk_stems):
            raise ValueError(
                f'{scene} chunk {chunk_idx}: predictor returned '
                f'{len(prediction.pred_pans)} maps for {len(chunk_stems)} stems')
        for stem, pred_pan in zip(chunk_stems, prediction.pred_pans):
            if pred_pan.dtype != np.int32 or pred_pan.shape != (h, w):
                raise ValueError(
                    f'{scene} stem {stem}: predictor returned pred_pan with '
                    f'dtype={pred_pan.dtype} shape={pred_pan.shape}, expected '
                    f'int32 {(h, w)}')
            try:
                validate_pred_pan(pred_pan, spec.label)
            except ValueError as e:
                raise ValueError(f'{scene} stem {stem}: {e}') from None

        gt_pans = [load_gt(scene_dir, s, spec.label) for s in chunk_stems]
        chunk = chunk_from_maps(
            prediction.pred_pans, gt_pans, chunk_stems,
            scene=f'{spec.dataset}__{scene}', chunk_idx=chunk_idx,
            view_offset=view_offset, n=spec.n)
        del prediction, gt_pans                        # keep one copy of the maps
        yield chunk


def _write_chunk(raw_dir: str, spec: CellSpec, scene: str, chunk: Chunk) -> None:
    """Write one chunk to its npz under ``raw_dir``, replacing an existing file."""
    write_chunk_npz(
        chunk_path(raw_dir, spec.dataset, scene, chunk.chunk_idx),
        list(chunk.pred_pan), list(chunk.gt_pan), chunk.stems,
        scene=chunk.scene, chunk_idx=chunk.chunk_idx,
        view_offset=chunk.view_offset, n=chunk.n, overwrite=True,
    )


def _scenes_or_raise(spec: CellSpec) -> list[str]:
    """``cell_scenes(spec)``; ValueError if it is empty, a warning if some
    requested scenes of the frame list are not on disk."""
    scenes = cell_scenes(spec)
    dataset_dir = os.path.join(spec.data_root, spec.dataset)
    if not scenes:
        raise ValueError(
            f'no scene of frame list {spec.frame_list!r} found under '
            f'{dataset_dir!r} (after scenes); '
            f'check data_root and dataset (e.g. scannetpp_spp100, scannetpp_spp100_iphone, '
            f'scannetv2_s20)')
    listed = load_frame_list(spec.frame_list)
    wanted = [s for s in listed if spec.scenes is None or s in spec.scenes]
    missing = sorted(set(wanted) - set(scenes))
    if missing:
        logger.warning('WARNING: %d of the %d scenes of frame list %r are not under %s and are '
                       'not scored (first: %s)', len(missing), len(wanted), spec.frame_list,
                       dataset_dir, ', '.join(missing[:5]))
    return scenes


def evaluate_cell(
    predictor: ChunkPredictor,
    spec: CellSpec,
    cfg: RecomputeConfig,
    dump_dir: str | None = None,
    overwrite: bool = False,
) -> tuple[dict, list[dict]]:
    """Predict and score every chunk of the cell; optionally also dump it.

    Scenes in ``cell_scenes(spec)`` order (sorted). Each scene's chunks go
    from ``iter_scene_chunks`` straight into ``vcpq.recompute.score_scene``
    (one chunk in memory at a time), and the scene results are merged by
    ``vcpq.recompute.aggregate_scenes``, the merge of ``vcpq-recompute
    --workers N``; the numbers equal those of ``vcpq-recompute`` on a dump
    of the same predictions. The predictor is called once per chunk. A scene
    without usable frames contributes nothing, as a scene without chunk files
    does in ``vcpq-recompute``.

    Args:
        predictor: The method.
        spec: The cell.
        cfg: Scoring configuration (``cfg.workers`` is not used).
        dump_dir: Also write every chunk under this directory
            (``<dataset>__<scene>/c*.npz``), so that ``vcpq-recompute`` can
            score it again later. None: nothing is written.
        overwrite: Replace chunk files that already exist under ``dump_dir``.

    Returns:
        (result, chunk_records): ``result`` has the ``recompute.json`` schema
        with ``raw_dir`` = the absolute ``dump_dir`` (None without it);
        ``chunk_records`` are the per-chunk records of
        ``recompute_chunks.json.gz``.

    Raises:
        ValueError: No scene of the frame list is a directory under
            ``{data_root}/{dataset}`` (wrong ``data_root`` / ``dataset``, or a
            frame list of another dataset), or none has a usable frame.
        FileExistsError: ``dump_dir`` already holds chunk files of this
            cell's scenes and ``overwrite`` is False; raised before the first
            prediction.
    """
    scenes = _scenes_or_raise(spec)
    if dump_dir is not None and not overwrite:
        existing = [p for scene in scenes
                    for p in sorted(glob.glob(os.path.join(dump_dir, f'{spec.dataset}__{scene}', 'c*.npz')))]
        if existing:
            raise FileExistsError(
                f'{len(existing)} chunk file(s) of this cell already exist under {dump_dir} '
                f'(first: {existing[0]})')
    tags: list[str] = []

    def dumped(chunks: Iterator[Chunk], scene: str) -> Iterator[Chunk]:
        for chunk in chunks:
            # No chunk file of the cell existed at the start (or overwrite):
            # every chunk is written, so the dump holds what was scored.
            _write_chunk(dump_dir, spec, scene, chunk)
            yield chunk

    def scene_results() -> Iterator[SceneResult]:
        for k, scene in enumerate(scenes, 1):
            chunks = iter_scene_chunks(predictor, spec, scene)
            if dump_dir is not None:
                chunks = dumped(chunks, scene)
            first = next(chunks, None)
            if first is None:                          # no usable frame
                continue
            result = score_scene(itertools.chain([first], chunks), cfg)
            tags.append(result.tag)
            logger.info('  scene %d/%d %s: %d chunks', k, len(scenes), result.tag, len(result.chunks))
            yield result

    blocks, chunk_records, n_void = aggregate_scenes(scene_results(), cfg)
    if not chunk_records:
        raise ValueError(f'no scene of frame list {spec.frame_list!r} has a usable frame '
                         f'(ground truth on both sides)')
    # one chunk record per chunk
    return cell_result(dump_dir, cfg, len(tags), len(chunk_records), n_void, blocks), chunk_records


class GroundTruthPredictor:
    """Test double: "predicts" the GT panoptic map itself — a perfect
    predictor, used to check the rest of the pipeline (e.g. VC-PQ == 1.0 on
    a recompute of its dump) independent of any real model."""

    def __init__(self, spec: CellSpec):
        self.spec = spec

    def predict(self, image_paths: list[str], stems: list[str],
                gt_size: tuple[int, int]) -> ChunkPrediction:
        # image_paths[i] == {scene_dir}/color/{stem}.<ext> -> scene_dir is two
        # path components up.
        scene_dir = os.path.dirname(os.path.dirname(image_paths[0]))
        pred_pans = [load_gt(scene_dir, stem, self.spec.label) for stem in stems]
        return ChunkPrediction(pred_pans=pred_pans)
