"""The scorer: chunks -> ``recompute.json`` (+ per-chunk records).

This is the scorer behind every VC-PQ / PQ^scene number in the paper.
``vcpq-eval`` (``vcpq.protocol.runner.evaluate_cell``) feeds it the chunks of
a predictor as they are predicted; ``vcpq-recompute`` (``main``) reads them
from a ``raw_chunks/`` directory (``vcpq.protocol.dump``: one
``<dataset__scene>/c*.npz`` per chunk holding the PRE-robust-filter pred / GT
panoptic maps), without any model forward. Either way it computes, by
default, the two metrics of the paper with class-wise matching:

  - VC-PQ / VC-SQ / VC-RQ with robust filtering off (robust_filter_off), for every
    (eta, gamma): class averages (total / things / stuff), the 1 - VC-PQ
    error decomposition, the per-scene mean and the pooled per-class raw
    counters;
  - PQ^scene / SQ / RQ (tiling) with robust filtering on (robust_filter_on), same layout
    (eta/gamma-independent).

``both_robust_filter`` adds the other robust-filtering state of each metric (VC-PQ
robust_filter_on, PQ^scene robust_filter_off); ``class_agnostic`` adds the class-agnostic and
things-only class-agnostic variants of every computed state.

Aggregation: one accumulator per physical scene collects
its chunks (begin_scene / end_scene around EVERY chunk); the per-scene value
is the mean over scenes; class-avg pools every chunk of every scene.

Three code paths give identical numbers:
  - ``workers == 1``: serial reference path, one pass over the cell per metric
    family (``recompute_pq_scene``, ``recompute_vcpq`` per (eta, gamma)).
    Writes no per-chunk records.
  - ``workers > 1``: scene-parallel single pass (``score_scene`` per scene in a
    process pool, merged by ``aggregate_scenes`` in sorted scene order). Also
    returns one record per chunk with the same raw counters restricted to that
    chunk (the accumulator deltas around the chunk's end_scene), written to
    ``recompute_chunks.json.gz``; integer fields sum exactly to ``per_class``,
    float sums agree to float rounding.
  - ``vcpq-eval``: ``score_scene`` per scene on the predictor's chunks, in
    sorted scene order, merged by the same ``aggregate_scenes``.

The JSON block builders live in ``vcpq.recompute_blocks``.

Notes on the output:
  - ``robust_filter_threshold`` is the threshold of VC-PQ's robust_filter_on variant (computed
    with ``both_robust_filter``); 0 makes it equal robust_filter_off. The protocol value and the CLI
    default ``--robust_filter_threshold`` are 0.005, and PQ^scene always uses 0.005;
  - a scene whose GT is entirely void in every chunk is left out of the
    per-scene means unless ``include_void_scenes``; ``n_void_scenes`` counts
    such scenes either way. The paper's per-scene numbers include these scenes
    (value 0); pass ``--include_void_scenes`` to reproduce them. Pooled
    (class-avg) numbers are unaffected.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import numpy as np

from vcpq.label_systems import LabelSystem, load_label_system
from vcpq.pq_scene_accumulator import PQSceneAccumulator
from vcpq.protocol.dump import Chunk, iter_cell, read_chunk_npz
from vcpq.recompute_blocks import pq_block, vcpq_block
from vcpq.vcpq_accumulator import VCPQAccumulator

logger = logging.getLogger('vcpq.recompute')

# Panoptic-Lifting robust-class filter threshold of the protocol; PQ^scene
# always uses it, VC-PQ uses RecomputeConfig.robust_filter_threshold.
PROTOCOL_RF = 0.005

# Stats variant name -> (raw_stats flags, accumulator attribute that merge() reads).
_VARIANTS = (
    ('robust_filter_off', dict(rf=False, class_agnostic=False, things_only=False), 'stats'),
    ('robust_filter_on', dict(rf=True, class_agnostic=False, things_only=False), 'stats_rf'),
    ('class_agnostic_robust_filter_off', dict(rf=False, class_agnostic=True, things_only=False), 'stats_ca'),
    ('class_agnostic_robust_filter_on', dict(rf=True, class_agnostic=True, things_only=False), 'stats_ca_rf'),
    ('class_agnostic_things_robust_filter_off', dict(rf=False, class_agnostic=True, things_only=True), 'stats_cath'),
    ('class_agnostic_things_robust_filter_on', dict(rf=True, class_agnostic=True, things_only=True), 'stats_cath_rf'),
)

# Header of recompute_chunks.json.gz.
CHUNK_FIELDS = {
    'vcpq': 'per (eta,gamma) x the computed variants among {robust_filter_off,robust_filter_on,class_agnostic_robust_filter_off,class_agnostic_robust_filter_on,'
            'class_agnostic_things_robust_filter_off,class_agnostic_things_robust_filter_on}: '
            '{class: {tp,fp,fn,q_sum,i_sum,c_sum,fp_void_skipped,fn_a,fn_b,fn_c,fp_a,fp_b,fp_c}}',
    'pq_scene': 'the computed variants: {class: {tp,fp,fn,iou_sum}}',
}


@dataclass
class RecomputeConfig:
    """What to recompute.

    Attributes:
        label: Label system of the cell.
        etas: VC-PQ visibility thresholds (px) to sweep.
        gammas: VC-PQ view-weighting exponents to sweep (0 = VC-PQ,
            1 = area-weighted).
        robust_filter_threshold: Robust-filter threshold of VC-PQ's robust_filter_on variant
            (``both_robust_filter``); 0 turns the filter off (robust_filter_on == robust_filter_off). PQ^scene
            always uses ``PROTOCOL_RF``.
        workers: 1 = serial reference path; > 1 = scene-parallel pool (also
            produces per-chunk records).
        include_void_scenes: Let scenes whose GT is entirely void enter the
            per-scene means (value 0, as in the paper's numbers). Default
            False: they are left out and counted in ``n_void_scenes``.
        class_agnostic: Also compute the class-agnostic variants (every
            segment relabelled to one class, and the same on things only).
        both_robust_filter: Also compute the other robust-filtering state of each
            metric: VC-PQ with the filter on, PQ^scene with it off.
    """

    label: LabelSystem
    etas: tuple[int, ...] = (100,)
    gammas: tuple[float, ...] = (0.0,)
    robust_filter_threshold: float = PROTOCOL_RF
    workers: int = 1
    include_void_scenes: bool = False
    class_agnostic: bool = False
    both_robust_filter: bool = False

    def __post_init__(self) -> None:
        # Normalised so the block keys read 'eta100_gamma0.0'.
        self.etas = tuple(int(e) for e in self.etas)
        self.gammas = tuple(float(g) for g in self.gammas)
        if self.workers < 1:
            raise ValueError(f'workers must be >= 1, got {self.workers}')


@dataclass
class SceneResult:
    """One scene's accumulated statistics (``score_scene``); picklable.

    Attributes:
        tag: Scene tag (``<dataset>__<scene>`` dir name).
        vc: (eta, gamma) -> computed variant (``robust_filter_off`` ... ``class_agnostic_things_robust_filter_on``)
            -> class -> raw VC-PQ counters.
        vc_per_scene: (eta, gamma) -> computed state (``robust_filter_off`` / ``robust_filter_on``)
            -> the scene's VC-PQ (class-avg within the scene, total).
        pq, pq_per_scene: Same for PQ^scene ({tp, fp, fn, iou_sum}; PQ).
        chunks: One record per chunk (``recompute_chunks.json.gz`` schema).
        all_void: The GT is entirely void in every chunk.
    """

    tag: str
    vc: dict[tuple[int, float], dict[str, dict[int, dict]]]
    vc_per_scene: dict[tuple[int, float], dict[str, float]]
    pq: dict[str, dict[int, dict]]
    pq_per_scene: dict[str, float]
    chunks: list[dict]
    all_void: bool


# ──────────────────────────────────────────────────────────────────────────────
# What is computed, and the accumulator factories
# ──────────────────────────────────────────────────────────────────────────────
def _rf_states(cfg: RecomputeConfig, family: str) -> tuple[tuple[str, bool], ...]:
    """(name, ``rf`` flag) of the robust-filtering states a family computes:
    the metric's own one (VC-PQ robust_filter_off, PQ^scene robust_filter_on), or both with ``both_robust_filter``."""
    if cfg.both_robust_filter:
        return (('robust_filter_off', False), ('robust_filter_on', True))
    return (('robust_filter_off', False),) if family == 'vcpq' else (('robust_filter_on', True),)


def _variants_of(cfg: RecomputeConfig, family: str) -> list[tuple[str, dict, str]]:
    """The ``_VARIANTS`` entries a family computes under ``cfg``."""
    rfs = {rf for _, rf in _rf_states(cfg, family)}
    return [v for v in _VARIANTS
            if v[1]['rf'] in rfs and (cfg.class_agnostic or not v[1]['class_agnostic'])]


def _robust_filter(cfg: RecomputeConfig, family: str) -> str:
    """The accumulators' ``robust_filter`` option for a family under ``cfg``."""
    if cfg.both_robust_filter:
        return 'both'
    return 'off' if family == 'vcpq' else 'on'


def _new_vc(cfg: RecomputeConfig, eta: int, gamma: float) -> VCPQAccumulator:
    return VCPQAccumulator(cfg.label, eta=eta, gamma=gamma, robust_filter_threshold=cfg.robust_filter_threshold,
                           class_agnostic=cfg.class_agnostic,
                           robust_filter=_robust_filter(cfg, 'vcpq'))


def _new_pq(cfg: RecomputeConfig) -> PQSceneAccumulator:
    return PQSceneAccumulator(cfg.label, robust_filter_threshold=PROTOCOL_RF, class_agnostic=cfg.class_agnostic,
                              robust_filter=_robust_filter(cfg, 'pq_scene'))


# ──────────────────────────────────────────────────────────────────────────────
# Serial reference path: one pass over the cell per metric family
# ──────────────────────────────────────────────────────────────────────────────
def _load(paths: list[str]) -> Iterator[Chunk]:
    return (read_chunk_npz(p) for p in paths)


def _paths_all_void(paths: list[str]) -> bool:
    """True if the GT of every chunk npz is entirely void (reads gt_pan only)."""
    for p in paths:
        with np.load(p) as d:
            if d['gt_pan'].any():
                return False
    return True


def _serial_family(scenes: list[tuple[str, list[str]]], new: Callable, scene_value: Callable,
                   rf_states: tuple[tuple[str, bool], ...], exclude: frozenset) -> tuple:
    """(pooled accumulator, computed state -> per-scene values)."""
    overall = new()
    ps: dict[str, list] = {name: [] for name, _ in rf_states}
    for tag, paths in scenes:
        acc = new()
        for chunk in _load(paths):
            acc.begin_scene()
            for pp, gp in zip(chunk.pred_pan, chunk.gt_pan):
                acc.add_frame(pp, gp)
            acc.end_scene()
        if tag not in exclude:
            for name, rf in rf_states:
                ps[name].append(scene_value(acc, rf))
        overall.merge(acc)
    return overall, ps


def recompute_vcpq(scenes: list[tuple[str, list[str]]], cfg: RecomputeConfig, eta: int,
                   gamma: float, exclude: frozenset = frozenset()) -> dict:
    """VC-PQ per-scene (mean, scenes in ``exclude`` left out) + class-avg (pooled)."""
    return vcpq_block(*_serial_family(
        scenes, lambda: _new_vc(cfg, eta, gamma),
        lambda a, rf: a.compute_metrics(rf=rf)['total']['vcpq'], _rf_states(cfg, 'vcpq'), exclude),
        class_agnostic=cfg.class_agnostic)


def recompute_pq_scene(scenes: list[tuple[str, list[str]]], cfg: RecomputeConfig,
                       exclude: frozenset = frozenset()) -> dict:
    """PQ^scene per-scene (mean) + class-avg (pooled)."""
    return pq_block(*_serial_family(
        scenes, lambda: _new_pq(cfg),
        lambda a, rf: a.compute_metrics(rf=rf)['total']['pq'], _rf_states(cfg, 'pq_scene'), exclude),
        class_agnostic=cfg.class_agnostic)


# ──────────────────────────────────────────────────────────────────────────────
# Scene-parallel path: one scene read once, every accumulator filled, plain
# stats returned; the parent merges them in sorted scene order. Merge is
# counter addition, so same order -> identical results to the serial path.
# ──────────────────────────────────────────────────────────────────────────────
def _snap(stats: dict) -> dict:
    """Copy of a {class: {field: count}} stats dict (insertion order kept)."""
    return {c: dict(s) for c, s in stats.items()}


def _delta(after: dict, before: dict) -> dict:
    """Per-class counts contributed by one chunk = after - before (classes with
    any non-zero contribution only; string keys for JSON)."""
    out = {}
    for c, s in after.items():
        b = before.get(c)
        d = {k: (v - b[k] if b is not None else v) for k, v in s.items()}
        if any(v != 0 for v in d.values()):
            out[str(c)] = d
    return out


def _variants(a, variants: list[tuple[str, dict, str]]) -> dict:
    """variant name -> copy of that stats dict, for a VC-PQ or PQ^scene accumulator."""
    return {name: _snap(a.raw_stats(**flags)) for name, flags, _ in variants}


class _Plain:
    """Plain-attribute stand-in for a merged-from accumulator (``merge`` reads
    only the stats attributes)."""

    def __init__(self, **stats):
        for k, v in stats.items():
            setattr(self, k, {c: dict(s) for c, s in v.items()})

    @classmethod
    def of_variants(cls, variants: dict) -> '_Plain':
        """From a ``_variants`` dict; variants not computed merge as empty."""
        return cls(**{attr: variants.get(name, {}) for name, _, attr in _VARIANTS})


def score_scene(chunks: Iterable[Chunk], cfg: RecomputeConfig, tag: str | None = None,
                chunk_names: Iterable[str] | None = None) -> SceneResult:
    """Score one scene: PQ^scene and VC-PQ per (eta, gamma), the variants of ``cfg``.

    One accumulator per family per (eta, gamma); begin_scene / end_scene
    around EVERY chunk. Chunks are consumed one at a time (pass a generator
    to keep one chunk in memory).

    Args:
        chunks: The scene's chunks in order.
        cfg: Recompute configuration.
        tag: Scene tag for the records (default: the first chunk's ``scene``).
        chunk_names: Record name per chunk (default ``c{chunk_idx:04d}``;
            the cell path passes the npz file stem).

    Returns:
        The scene's raw statistics, per-scene values and per-chunk records.
    """
    pq = _new_pq(cfg)
    vc = {(e, g): _new_vc(cfg, e, g) for e in cfg.etas for g in cfg.gammas}
    pq_vars, vc_vars = _variants_of(cfg, 'pq_scene'), _variants_of(cfg, 'vcpq')
    names = iter(chunk_names) if chunk_names is not None else None
    records: list[dict] = []
    all_void = True
    for chunk in chunks:
        tag = chunk.scene if tag is None else tag
        name = next(names) if names is not None else f'c{chunk.chunk_idx:04d}'
        all_void = all_void and not chunk.gt_pan.any()
        pq.begin_scene()
        for a in vc.values():
            a.begin_scene()
        for pp, gp in zip(chunk.pred_pan, chunk.gt_pan):
            pq.add_frame(pp, gp)
            for a in vc.values():
                a.add_frame(pp, gp)
        # Snapshot every counter before this chunk's end_scene: the per-chunk
        # raw statistics are the deltas (the accumulators are untouched).
        pq0 = _variants(pq, pq_vars)
        vc0 = {key: _variants(a, vc_vars) for key, a in vc.items()}
        pq.end_scene()
        # The gamma variants see identical frames: share one scene cache so the
        # robust-filtered / things-only maps and step-2 overlap tables are built once.
        shared: dict = {}
        for a in vc.values():
            a.scene_cache = shared
            a.end_scene()
            a.scene_cache = None
        shared.clear()
        records.append({
            'scene': tag, 'chunk': name, 'n_frames': int(len(chunk.pred_pan)),
            'vcpq': {f'eta{e}_gamma{g}': {n: _delta(a.raw_stats(**flags), vc0[(e, g)][n])
                                          for n, flags, _ in vc_vars}
                     for (e, g), a in vc.items()},
            'pq_scene': {n: _delta(pq.raw_stats(**flags), pq0[n]) for n, flags, _ in pq_vars},
        })
    if not records:
        raise ValueError('score_scene: no chunks')
    return SceneResult(
        tag=tag,
        vc={key: _variants(a, vc_vars) for key, a in vc.items()},
        vc_per_scene={key: {name: a.compute_metrics(rf=rf)['total']['vcpq']
                            for name, rf in _rf_states(cfg, 'vcpq')}
                      for key, a in vc.items()},
        pq=_variants(pq, pq_vars),
        pq_per_scene={name: pq.compute_metrics(rf=rf)['total']['pq']
                      for name, rf in _rf_states(cfg, 'pq_scene')},
        chunks=records,
        all_void=all_void,
    )


def _scene_worker(job: tuple) -> SceneResult:
    """Pool worker: load one scene's chunks lazily and score them."""
    tag, paths, cfg = job
    names = [os.path.basename(p)[:-4] for p in paths]
    return score_scene(_load(paths), cfg, tag=tag, chunk_names=names)


def aggregate_scenes(results: Iterable[SceneResult], cfg: RecomputeConfig,
                     n_total: int | None = None) -> tuple[dict, list[dict], int]:
    """Merge per-scene results, in the order given, into the family blocks.

    The merge shared by the scene-parallel recompute and by ``vcpq-eval``
    (``vcpq.protocol.runner.evaluate_cell``); give the scenes in sorted order
    to match the serial path.

    Args:
        results: ``score_scene`` results, one per scene.
        cfg: The configuration the results were computed with.
        n_total: Number of scenes, for the progress log (None: no log).

    Returns:
        (family blocks {'pq_scene', 'vcpq'}, chunk records, number of
        all-void-GT scenes).
    """
    pq_all = _new_pq(cfg)
    vc_all = {(e, g): _new_vc(cfg, e, g) for e in cfg.etas for g in cfg.gammas}
    pq_ps = []
    vc_ps: dict = {k: [] for k in vc_all}
    chunk_records: list[dict] = []
    n_void = 0
    for i, r in enumerate(results, 1):
        chunk_records.extend(r.chunks)
        pq_all.merge(_Plain.of_variants(r.pq))
        for k, a in vc_all.items():
            a.merge(_Plain.of_variants(r.vc[k]))
        n_void += r.all_void
        if cfg.include_void_scenes or not r.all_void:
            pq_ps.append(r.pq_per_scene)
            for k in vc_all:
                vc_ps[k].append(r.vc_per_scene[k])
        if n_total and (i % 10 == 0 or i == n_total):
            logger.info('  scenes %d/%d', i, n_total)

    def cols(ps: list, family: str) -> dict[str, list]:
        return {name: [x[name] for x in ps] for name, _ in _rf_states(cfg, family)}

    blocks = {
        'pq_scene': pq_block(pq_all, cols(pq_ps, 'pq_scene'), class_agnostic=cfg.class_agnostic),
        'vcpq': {f'eta{e}_gamma{g}': vcpq_block(a, cols(vc_ps[(e, g)], 'vcpq'),
                                                class_agnostic=cfg.class_agnostic)
                 for (e, g), a in vc_all.items()},
    }
    return blocks, chunk_records, n_void


def recompute_parallel(scenes: list[tuple[str, list[str]]], cfg: RecomputeConfig
                       ) -> tuple[dict, list[dict], int]:
    """All metric families in one scene pass, scene-parallel; same blocks as the serial path.

    Returns:
        (family blocks {'pq_scene', 'vcpq'}, chunk records, number of
        all-void-GT scenes).
    """
    jobs = [(tag, paths, cfg) for tag, paths in scenes]
    with mp.get_context('fork').Pool(cfg.workers) as pool:
        # imap keeps job order
        return aggregate_scenes(pool.imap(_scene_worker, jobs), cfg, n_total=len(jobs))


# ──────────────────────────────────────────────────────────────────────────────
# Cell entry points
# ──────────────────────────────────────────────────────────────────────────────
def recompute_cell(raw_dir: str, cfg: RecomputeConfig) -> tuple[dict, list[dict] | None]:
    """Recompute every metric family of one cell.

    Args:
        raw_dir: A ``raw_chunks/`` directory (``<dataset__scene>/c*.npz``).
        cfg: Recompute configuration.

    Returns:
        (result, chunk_records). ``result`` has the ``recompute.json`` schema
        (``cell_result``; ``chunks_file`` is added by ``write_recompute``).
        ``chunk_records`` is None on the serial path (``workers == 1``).

    Raises:
        ValueError: ``raw_dir`` holds no scene directory with chunk npz files.
    """
    scenes = [(tag, paths) for tag, paths in iter_cell(raw_dir) if paths]
    if not scenes:
        raise ValueError(f'no <dataset__scene>/c*.npz under {raw_dir}')
    n_chunks = sum(len(paths) for _, paths in scenes)
    logger.info('Loaded %d scenes / %d chunks from %s', len(scenes), n_chunks, raw_dir)
    if cfg.workers > 1:
        blocks, chunk_records, n_void = recompute_parallel(scenes, cfg)
    else:
        void = frozenset(tag for tag, paths in scenes if _paths_all_void(paths))
        exclude = frozenset() if cfg.include_void_scenes else void
        blocks = {
            'pq_scene': recompute_pq_scene(scenes, cfg, exclude),   # eta/gamma-independent
            'vcpq': {f'eta{e}_gamma{g}': recompute_vcpq(scenes, cfg, e, g, exclude)
                     for e in cfg.etas for g in cfg.gammas},
        }
        chunk_records, n_void = None, len(void)
    return cell_result(raw_dir, cfg, len(scenes), n_chunks, n_void, blocks), chunk_records


def cell_result(raw_dir: str | None, cfg: RecomputeConfig, n_scenes: int, n_chunks: int,
                n_void: int, blocks: dict) -> dict:
    """The ``recompute.json`` dict of one cell.

    Keys: ``raw_dir`` (absolute path of the chunk directory, None when the
    chunks were scored without being saved), ``label_system``, ``n_scenes``,
    ``n_chunks``, ``n_void_scenes``, ``include_void_scenes`` (whether those
    scenes entered the per-scene means), ``robust_filter_threshold`` (the robust-filter
    threshold of VC-PQ's robust_filter_on variant), then the family blocks
    (``pq_scene``, ``vcpq``).
    """
    return {
        'raw_dir': None if raw_dir is None else os.path.abspath(raw_dir),
        'label_system': cfg.label.name,
        'n_scenes': n_scenes, 'n_chunks': n_chunks,
        'n_void_scenes': n_void,
        'include_void_scenes': cfg.include_void_scenes,
        'robust_filter_threshold': cfg.robust_filter_threshold,
        **blocks,
    }


def write_recompute(out_json: str | None, result: dict, chunk_records: list[dict] | None,
                    chunks_out: str | None = None, *, cfg: RecomputeConfig | None = None) -> str | None:
    """Write ``recompute.json`` and, given chunk records, ``recompute_chunks.json.gz``.

    Args:
        out_json: JSON path (None -> print the JSON to stdout).
        result: ``recompute_cell`` result (not modified).
        chunk_records: ``recompute_cell`` chunk records; None -> no chunks file.
        chunks_out: Chunks file path (default ``<dir of out_json>/recompute_chunks.json.gz``;
            with neither this nor ``out_json`` no chunks file is written).
            ``result['chunks_file']`` stores it relative to the JSON's directory.
        cfg: The configuration ``result`` was computed with (etas / gammas /
            robust_filter_threshold of the chunks-file header); required with chunk records.

    Returns:
        The chunks file path, or None if none was written.
    """
    result = dict(result)
    chunks_path = None
    if chunk_records is not None and (out_json or chunks_out):
        if cfg is None:
            raise ValueError('write_recompute: cfg is required to write chunk records')
        chunks_path = chunks_out or os.path.join(os.path.dirname(out_json) or '.',
                                                 'recompute_chunks.json.gz')
        # Relative to the JSON's directory (cells move between machines).
        result['chunks_file'] = os.path.relpath(os.path.abspath(chunks_path),
                                                os.path.dirname(os.path.abspath(out_json or chunks_path)))
    if out_json:
        os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))
    if chunks_path:
        os.makedirs(os.path.dirname(chunks_path) or '.', exist_ok=True)
        payload = {
            'raw_dir': result['raw_dir'], 'label_system': result['label_system'],
            'etas': list(cfg.etas), 'gammas': list(cfg.gammas), 'robust_filter_threshold': cfg.robust_filter_threshold,
            'n_chunks': len(chunk_records),
            'fields': CHUNK_FIELDS,
            'chunks': chunk_records,
        }
        with gzip.open(chunks_path, 'wt', compresslevel=6) as f:
            json.dump(payload, f, separators=(',', ':'))
    return chunks_path


def add_scoring_args(ap: argparse.ArgumentParser) -> None:
    """The scoring options shared by ``vcpq-recompute`` and ``vcpq-eval``."""
    ap.add_argument('--gammas', type=float, nargs='+', default=[0.0],
                    help='VC-PQ view-weighting exponents to sweep (0 = VC-PQ, 1 = area-weighted)')
    ap.add_argument('--etas', type=int, nargs='+', default=[100], help='VC-PQ visibility px')
    ap.add_argument('--robust_filter_threshold', type=float, default=PROTOCOL_RF,
                    help=f'threshold of the robust filter in the VC-PQ variant with the filter on '
                         f'(--both_robust_filter), applied to VC-PQ only (PQ^scene always uses '
                         f'{PROTOCOL_RF}); 0 turns the filter off (default: {PROTOCOL_RF}, the '
                         f'protocol value)')
    ap.add_argument('--both_robust_filter', action='store_true', default=False,
                    help='also compute the other robust-filtering state of each metric (VC-PQ with '
                         'the filter on, PQ^scene with it off); by default VC-PQ is computed with '
                         'the filter off and PQ^scene with it on, as in the paper')
    ap.add_argument('--class_agnostic', action='store_true', default=False,
                    help='also compute the class-agnostic variants (every segment relabelled to one '
                         'class, and the same on things only); by default matching is class-wise')
    ap.add_argument('--include_void_scenes', action='store_true', default=False,
                    help='let scenes whose GT is entirely void enter the per-scene means with '
                         'their value (0), as in the paper\'s numbers; default: leave them out '
                         '(counted in n_void_scenes)')


def scoring_config(args: argparse.Namespace, workers: int = 1) -> RecomputeConfig:
    """``RecomputeConfig`` from ``--label_system`` and the ``add_scoring_args`` options."""
    return RecomputeConfig(load_label_system(args.label_system), etas=tuple(args.etas),
                           gammas=tuple(args.gammas),
                           robust_filter_threshold=args.robust_filter_threshold, workers=workers,
                           include_void_scenes=args.include_void_scenes,
                           class_agnostic=args.class_agnostic, both_robust_filter=args.both_robust_filter)


def _pct(x: float) -> str:
    return f'{x * 100:.1f}'


def _state(key: str) -> str:
    """'robust_filter_off' -> 'robust filtering off' (for the printed summary)."""
    return key.replace('robust_filter_', 'robust filtering ')


def print_summary(result: dict) -> None:
    """Print the headline numbers of a ``recompute.json`` result (or of a
    ``vcpq-pool`` one), for every computed robust-filtering state; the paper's
    numbers are VC-PQ (eta 100, gamma 0) with robust filtering off and PQ^scene
    with robust filtering on."""
    print(f'  scored: {result["n_scenes"]} scenes, {result["n_chunks"]} chunks')
    for key, v in result['vcpq'].items():
        for state, groups in v['class_avg'].items():
            t = groups['total']
            paper = key == 'eta100_gamma0.0' and state == 'robust_filter_off'
            name = f'{"paper VC-PQ, " if paper else ""}class-avg, {_state(state)}'
            print(f'  VC-PQ {key} ({name}): {_pct(t["vcpq"])}  VC-SQ {_pct(t["vcsq"])}  '
                  f'VC-RQ {_pct(t["vcrq"])}  per-scene VC-PQ {_pct(v["per_scene"][state])}')
            print(f'    error decomposition: 1 - VC-PQ = acct_mask (eps_IoU) {_pct(t["acct_mask"])} '
                  f'+ acct_vc (eps_VC) {_pct(t["acct_vc"])} + acct_thr (eps_q) {_pct(t["acct_thr"])}')
    ps = result['pq_scene']
    for state, value in ps['per_scene'].items():
        paper = state == 'robust_filter_on'
        name = f'{"paper PQ^scene, " if paper else ""}per-scene mean, {_state(state)}'
        print(f'  PQ-Scene ({name}): {_pct(value)}')
    print('  PQ-Scene secondary, pooled class-avg: '
          + ', '.join(f'{_state(state)} {_pct(g["total"]["pq"])}' for state, g in ps['class_avg'].items()))
    print(f'  all-void-GT scenes: {result["n_void_scenes"]} '
          f'({"included in" if result["include_void_scenes"] else "left out of"} per-scene means)')


def main(argv: list[str] | None = None) -> None:
    """``vcpq-recompute``: recompute one cell's metrics from its raw chunk dump."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('raw_dir', help='the raw_chunks/ directory of a cell dump')
    ap.add_argument('--label_system', required=True, choices=['spp100', 'scannetv2_s20'])
    add_scoring_args(ap)
    ap.add_argument('--workers', type=int, default=1,
                    help='>1: scene-parallel single pass (each scene read once, all metrics, '
                         'per-chunk records); 1: serial reference path (one pass per metric family)')
    ap.add_argument('--out', type=str, default=None, help='write the JSON here (else stdout)')
    ap.add_argument('--chunks_out', type=str, default=None,
                    help='per-chunk raw statistics (gzip JSON; --workers >1 only). '
                         'Default: <dir of --out>/recompute_chunks.json.gz')
    ap.add_argument('--no_chunks', action='store_true', default=False,
                    help='do not write the per-chunk statistics file')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    cfg = scoring_config(args, workers=args.workers)
    result, chunk_records = recompute_cell(args.raw_dir, cfg)
    print_summary(result)
    if chunk_records is None and not args.no_chunks:
        print('  per-chunk statistics: skipped (serial reference path; use --workers >1)')

    chunks_path = write_recompute(args.out, result, None if args.no_chunks else chunk_records,
                                  args.chunks_out, cfg=cfg)
    if args.out:
        print(f'Written {args.out}')
    if chunks_path:
        print(f'Written {chunks_path} ({len(chunk_records)} chunks, '
              f'{os.path.getsize(chunks_path)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
