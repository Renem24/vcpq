"""PQ (Panoptic Quality) metric accumulators.

Provides:
- match_panoptic_segments: Core IoU > 0.5 matching (shared logic).
- compute_pq_from_stats: Compute PQ/SQ/RQ from accumulated stats.
- PQSceneAccumulator: Scene-level PQ (stack all frames, match once).

COCO Panoptic-style void handling:
- Void pixels (gt_pan == 0) excluded from union.
- Predicted segments mostly overlapping void are not FP.

Class averaging keeps the Panoptic Lifting rule: every class present in the
stats enters the mean, and a class with tp + fp + fn == 0 would enter as 0.0.
For PQ^scene the rule is moot: the accumulators here create a class key only
by incrementing its tp, fp or fn, so every class they produce has
tp + fp + fn >= 1 (a void-majority prediction that is not counted as FP
creates no key). VC-PQ, which does create such classes, leaves them out
(panopticapi rule, see VCPQAccumulator._compute).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np

from .label_systems import LabelSystem
from .overlap import panoptic_ids, panoptic_overlap_table
from .panoptic import SEM_MULT
from .robust_filter import RF_PASSES, apply_robust_filter, check_rf_variant

# Sentinel class id for the class-agnostic pass: every segment is relabelled to
# this single dummy class so matching ignores the class head entirely.
_CA_CLASS = -1


def _collapse_cats(segments: list[dict]) -> list[dict]:
    """Return segment dicts with every category_id set to the class-agnostic dummy.

    Instance ids (``id``) are kept, so each segment stays a distinct instance;
    only its class is dropped, which makes downstream matching class-agnostic.
    """
    return [dict(s, category_id=_CA_CLASS) for s in segments]


def _void_stuff(pan_map: np.ndarray, thing_ids: frozenset[int]) -> np.ndarray:
    """Return a copy of ``pan_map`` with every non-thing (stuff) pixel set to void.

    Things-only class-agnostic pass: stuff is dropped from BOTH pred and GT, so
    the maps are scored as instance segmentation (SegVGGT/ScanNet convention:
    GT stuff -> ignore). Predicted thing pixels landing on former GT stuff then
    fall under the ordinary void rule (removed from IoU; FP skipped if mostly void).
    """
    out = pan_map.copy()
    sem = out // SEM_MULT
    out[(sem != 0) & ~np.isin(sem, list(thing_ids))] = 0
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Shared matching logic
# ──────────────────────────────────────────────────────────────────────────────
def match_panoptic_segments(
    pred_pan: np.ndarray,
    pred_segments: list[dict],
    gt_pan: np.ndarray,
    gt_segments: list[dict],
    stats: Dict[int, dict],
    void_threshold: float = 0.5,
) -> None:
    """Greedy IoU > 0.5 matching between pred and GT panoptic segments.

    Matches are resolved greedily by descending IoU. Matched pairs contribute
    to TP; unmatched predictions (not mostly void) become FP; unmatched GT
    become FN. All counts are accumulated into ``stats`` in-place.

    Args:
        pred_pan: [H, W] predicted panoptic map (sem_id * 1000 + inst_id).
        pred_segments: List of predicted segment dicts {id, category_id, isthing}.
        gt_pan: [H, W] GT panoptic map.
        gt_segments: List of GT segment dicts.
        stats: Mutable dict mapping category_id -> {tp, fp, fn, iou_sum}.
            Updated in-place.
        void_threshold: Fraction of pred segment overlapping void above which
            the segment is not counted as FP (default: 0.5).
    """
    gt_matched: set = set()
    pred_matched: set = set()

    # One overlap table replaces the per-pair `pred_pan == pid` / `gt_pan == gid`
    # masks; the loops below are unchanged so ties still break in the same order.
    table = panoptic_overlap_table(
        pred_pan, gt_pan,
        [p['id'] for p in pred_segments], [g['id'] for g in gt_segments],
    )
    pred_info: dict = {
        pseg['id']: (int(table.pred_area[pi]), int(table.pred_void[pi]))
        for pi, pseg in enumerate(pred_segments) if table.pred_area[pi] > 0
    }

    # Compute IoU for all (pred, gt) pairs sharing same class
    matches: list = []
    for pi, pseg in enumerate(pred_segments):
        pid = pseg['id']
        pcat = pseg['category_id']
        if pid not in pred_info:
            continue
        parea, pred_void = pred_info[pid]

        for gi, gseg in enumerate(gt_segments):
            gid = gseg['id']
            gcat = gseg['category_id']
            if pcat != gcat:
                continue
            inter = int(table.inter[gi, pi])
            if inter == 0:
                continue
            union = (parea - pred_void) + int(table.gt_area[gi]) - inter
            if union <= 0:
                continue
            iou = inter / union
            if iou > 0.5:
                matches.append((iou, pid, gid, pcat))

    # Greedy matching: sort by IoU descending
    matches.sort(key=lambda x: -x[0])
    for iou, pid, gid, cat in matches:
        if pid in pred_matched or gid in gt_matched:
            continue
        pred_matched.add(pid)
        gt_matched.add(gid)
        stats[cat]['tp'] += 1
        stats[cat]['iou_sum'] += iou

    # FP: unmatched predictions (skip if mostly overlapping with void)
    for pseg in pred_segments:
        pid = pseg['id']
        if pid in pred_matched or pid not in pred_info:
            continue
        parea, pred_void = pred_info[pid]
        void_frac = pred_void / parea if parea > 0 else 0.0
        if void_frac > void_threshold:
            continue
        stats[pseg['category_id']]['fp'] += 1

    # FN: unmatched GT
    for gseg in gt_segments:
        if gseg['id'] not in gt_matched:
            stats[gseg['category_id']]['fn'] += 1


def compute_pq_from_stats(stats: Dict[int, dict], label: LabelSystem) -> dict:
    """Compute PQ/SQ/RQ per-class and for total/things/stuff.

    Every class key of ``stats`` enters the class means; a class with
    tp + fp + fn == 0 would enter as 0.0 (Panoptic Lifting rule), but the
    accumulators of this module never create one (see the module docstring).

    Args:
        stats: Dict mapping category_id -> {tp, fp, fn, iou_sum}.
        label: Label system; its ``thing_ids`` / ``stuff_ids`` select the
            classes of the things / stuff averages.

    Returns:
        Dict with keys 'per_class', 'total', 'things', 'stuff'.
    """
    thing_ids = label.thing_ids
    stuff_ids = label.stuff_ids

    per_class: dict = {}
    for cat in sorted(stats.keys()):
        s = stats[cat]
        tp, fp, fn = s['tp'], s['fp'], s['fn']
        denom = tp + 0.5 * fp + 0.5 * fn
        pq = s['iou_sum'] / denom if denom > 0 else 0.0
        sq = s['iou_sum'] / tp if tp > 0 else 0.0
        rq = tp / denom if denom > 0 else 0.0
        per_class[cat] = dict(pq=pq, sq=sq, rq=rq, tp=tp, fp=fp, fn=fn,
                              iou_sum=s['iou_sum'])

    def _avg(ids: set) -> dict:
        vals = [per_class[c] for c in ids if c in per_class]
        if not vals:
            return dict(pq=0, sq=0, rq=0)
        return dict(
            pq=np.mean([v['pq'] for v in vals]),
            sq=np.mean([v['sq'] for v in vals]),
            rq=np.mean([v['rq'] for v in vals]),
        )

    all_ids = set(per_class.keys())
    return dict(
        per_class=per_class,
        total=_avg(all_ids),
        things=_avg(all_ids & thing_ids),
        stuff=_avg(all_ids & stuff_ids),
    )


def _segments_from_panoptic_map(
    pan_map: np.ndarray,
    thing_ids: frozenset[int],
) -> list[dict]:
    """Extract segment dicts from a panoptic map.

    Same list as ``panoptic.segments_from_pan``, but histogram-based
    (``panoptic_ids``) instead of sorting, which matters on the stacked
    multi-view map.

    Args:
        pan_map: [H, W] panoptic IDs (sem_id * 1000 + inst_id).
        thing_ids: Set of thing class IDs.

    Returns:
        List of segment dicts {id, category_id, isthing}.
    """
    segments = []
    for pan_id in panoptic_ids(pan_map):
        sem_id = int(pan_id // SEM_MULT)
        segments.append(dict(
            id=int(pan_id), category_id=sem_id,
            isthing=sem_id in thing_ids,
        ))
    return segments


def _merge_stats(dst: Dict[int, dict], src: Dict[int, dict]) -> None:
    """Merge per-class stats from src into dst in-place."""
    for cat, s in src.items():
        dst[cat]['tp'] += s['tp']
        dst[cat]['fp'] += s['fp']
        dst[cat]['fn'] += s['fn']
        dst[cat]['iou_sum'] += s['iou_sum']


# ──────────────────────────────────────────────────────────────────────────────
# PQ-Scene accumulator (scene-level matching)
# ──────────────────────────────────────────────────────────────────────────────
class PQSceneAccumulator:
    """Scene-level PQ accumulator (PQ-Scene, the paper's PQ^scene).

    Collects per-frame panoptic maps, stacks them at scene/chunk end,
    and performs matching once on the stacked result. Requires cross-view
    consistent prediction IDs.

    Usage::

        accum = PQSceneAccumulator(label=load_label_system('spp100'))
        accum.begin_scene()
        for frame in frames:
            accum.add_frame(pred_pan, gt_pan)
        accum.end_scene()
        metrics = accum.compute_metrics()

    Every ``end_scene`` accumulates the stats variants that
    ``robust_filter`` selects — robust_filter_off (original maps), robust_filter_on (after the
    Panoptic-Lifting robust-class filter) or both — and, with
    ``class_agnostic``, two class-agnostic ones per selected variant.
    ``compute_metrics`` and ``raw_stats`` select a variant with their ``rf`` /
    ``class_agnostic`` / ``things_only`` flags. PQ^scene itself is robust_filter_on.

    Args:
        label: Label system; its ``thing_ids`` / ``stuff_ids`` define the
            things/stuff averages and the things-only class-agnostic pass.
        void_threshold: Fraction of a prediction's area on GT void above
            which an unmatched prediction is not counted as FP. Default: 0.5.
        robust_filter_threshold: Robust-class filter threshold of the robust_filter_on variant: a
            class occupying < robust_filter_threshold of the non-void pixels of the
            whole chunk, in pred OR GT, is voided before matching.
            Default: 0.005 (Panoptic Lifting). 0 disables the filter
            (robust_filter_on == robust_filter_off).
        class_agnostic: Also run the class-agnostic passes (every segment
            relabelled to one dummy class). Default: False.
        robust_filter: Which variants to accumulate: ``'off'`` (robust_filter_off
            only), ``'on'`` (robust_filter_on only, PQ^scene) or ``'both'``. Default:
            ``'both'``.

    Attributes:
        pred_pans, gt_pans: The current chunk's buffered [H, W] maps, one per
            view (public, so a caller can read them before ``end_scene``).
            Reset by ``begin_scene`` and ``end_scene``.
    """

    def __init__(
        self,
        label: LabelSystem,
        void_threshold: float = 0.5,
        robust_filter_threshold: float = 0.005,
        class_agnostic: bool = False,
        robust_filter: str = 'both',
    ) -> None:
        if robust_filter not in RF_PASSES:
            raise ValueError(f"robust_filter must be 'off', 'on' or 'both', got {robust_filter!r}")
        self.robust_filter = robust_filter
        self._run_rf_off, self._run_rf_on = RF_PASSES[robust_filter]
        self.stats: Dict[int, dict] = defaultdict(
            lambda: dict(tp=0, fp=0, fn=0, iou_sum=0.0)
        )
        # Unfiltered (robust_filter_off -> self.stats) and robust-filtered (robust_filter_on ->
        # self.stats_rf) matches, each run when robust_filter selects it.
        self.stats_rf: Dict[int, dict] = defaultdict(
            lambda: dict(tp=0, fp=0, fn=0, iou_sum=0.0)
        )
        # Class-agnostic pass (off by default): when on, end_scene matches two more
        # times with every segment relabelled to _CA_CLASS, removing the class head
        # from matching. robust_filter_off -> stats_ca, robust_filter_on -> stats_ca_rf.
        self.class_agnostic = class_agnostic
        self.stats_ca: Dict[int, dict] = defaultdict(
            lambda: dict(tp=0, fp=0, fn=0, iou_sum=0.0)
        )
        self.stats_ca_rf: Dict[int, dict] = defaultdict(
            lambda: dict(tp=0, fp=0, fn=0, iou_sum=0.0)
        )
        # Things-only class-agnostic pass: stuff voided in pred AND GT,
        # remaining things collapsed to _CA_CLASS. robust_filter_off -> stats_cath, robust_filter_on -> stats_cath_rf.
        self.stats_cath: Dict[int, dict] = defaultdict(
            lambda: dict(tp=0, fp=0, fn=0, iou_sum=0.0)
        )
        self.stats_cath_rf: Dict[int, dict] = defaultdict(
            lambda: dict(tp=0, fp=0, fn=0, iou_sum=0.0)
        )
        self.void_threshold = void_threshold
        self.label = label
        self.thing_ids = label.thing_ids
        self.stuff_ids = label.stuff_ids
        # robust_filter_on = Panoptic-Lifting robust-class filter: a class occupying < this
        # fraction of non-void pixels (pred OR GT) of a chunk is voided before
        # matching. 0 -> nothing is voided (robust_filter_on == robust_filter_off).
        self.robust_filter_threshold = robust_filter_threshold
        self.pred_pans: list[np.ndarray] = []
        self.gt_pans: list[np.ndarray] = []

    def begin_scene(self) -> None:
        """Call before adding frames of a new scene/chunk."""
        self.pred_pans = []
        self.gt_pans = []

    def add_frame(self, pred_pan: np.ndarray, gt_pan: np.ndarray) -> None:
        """Add one frame's panoptic maps to the current scene.

        Args:
            pred_pan: [H, W] predicted panoptic map with cross-view consistent IDs.
            gt_pan: [H, W] GT panoptic map with scene-consistent IDs.
        """
        self.pred_pans.append(pred_pan)
        self.gt_pans.append(gt_pan)

    def end_scene(self) -> None:
        """Stack all frames and perform scene-level PQ matching.

        Matches unfiltered (-> self.stats, robust_filter_off) and / or robust-filtered
        (-> self.stats_rf, robust_filter_on) maps, as ``robust_filter`` selects. Resets
        the frame buffer.
        """
        if not self.pred_pans:
            return

        if self._run_rf_off:
            # Stack all frames vertically into a single large image
            pred_stacked = np.vstack(self.pred_pans)   # [S*H, W]
            gt_stacked = np.vstack(self.gt_pans)        # [S*H, W]

            # robust_filter_off: unfiltered scene-level match.
            seg_p = _segments_from_panoptic_map(pred_stacked, self.thing_ids)
            seg_g = _segments_from_panoptic_map(gt_stacked, self.thing_ids)
            match_panoptic_segments(
                pred_stacked, seg_p, gt_stacked, seg_g,
                self.stats, self.void_threshold,
            )

            # class-agnostic robust_filter_off: same maps, class collapsed -> stats_ca.
            if self.class_agnostic:
                match_panoptic_segments(
                    pred_stacked, _collapse_cats(seg_p), gt_stacked, _collapse_cats(seg_g),
                    self.stats_ca, self.void_threshold,
                )
                # things-only: stuff voided in both maps, things collapsed -> stats_cath.
                self._match_things_only(pred_stacked, gt_stacked, self.stats_cath)

        if not self._run_rf_on:
            self.pred_pans = []
            self.gt_pans = []
            return

        # robust_filter_on: robust-filtered (Panoptic-Lifting filter: classes < 0.5% -> void), same match.
        # The filter pools the per-view maps of the chunk (same result as on
        # the stacked map); its output is stacked like the original maps.
        pf_views, gf_views = apply_robust_filter(self.pred_pans, self.gt_pans, self.robust_filter_threshold)
        pf, gf = np.vstack(pf_views), np.vstack(gf_views)
        seg_pf = _segments_from_panoptic_map(pf, self.thing_ids)
        seg_gf = _segments_from_panoptic_map(gf, self.thing_ids)
        match_panoptic_segments(
            pf, seg_pf, gf, seg_gf, self.stats_rf, self.void_threshold,
        )

        # class-agnostic robust_filter_on: robust-filtered maps, class collapsed -> stats_ca_rf.
        if self.class_agnostic:
            match_panoptic_segments(
                pf, _collapse_cats(seg_pf), gf, _collapse_cats(seg_gf),
                self.stats_ca_rf, self.void_threshold,
            )
            self._match_things_only(pf, gf, self.stats_cath_rf)

        self.pred_pans = []
        self.gt_pans = []

    def _match_things_only(
        self, pred_map: np.ndarray, gt_map: np.ndarray, stats: Dict[int, dict]
    ) -> None:
        """Things-only class-agnostic match of one (pred, gt) stacked pair into ``stats``."""
        thing_ids = self.thing_ids
        pt, gt_t = _void_stuff(pred_map, thing_ids), _void_stuff(gt_map, thing_ids)
        match_panoptic_segments(
            pt, _collapse_cats(_segments_from_panoptic_map(pt, thing_ids)),
            gt_t, _collapse_cats(_segments_from_panoptic_map(gt_t, thing_ids)),
            stats, self.void_threshold,
        )

    def merge(self, other: 'PQSceneAccumulator') -> None:
        """Merge another accumulator's stats into this one (all variants)."""
        _merge_stats(self.stats, other.stats)
        _merge_stats(self.stats_rf, other.stats_rf)
        _merge_stats(self.stats_ca, other.stats_ca)
        _merge_stats(self.stats_ca_rf, other.stats_ca_rf)
        _merge_stats(self.stats_cath, other.stats_cath)
        _merge_stats(self.stats_cath_rf, other.stats_cath_rf)

    def raw_stats(self, rf: bool = False, class_agnostic: bool = False,
                  things_only: bool = False) -> Dict[int, dict]:
        """The accumulated per-class stats dict of one variant (not a copy).

        {cat: {tp, fp, fn, iou_sum}}

        Args:
            rf: robust-filtered maps (robust_filter_on) instead of the original ones.
            class_agnostic: class head removed from matching (every segment
                collapsed to one dummy class). Empty unless the accumulator
                was built with class_agnostic=True.
            things_only: class-agnostic pass on maps whose stuff pixels are
                voided (pred AND GT); requires class_agnostic=True.

        Raises:
            ValueError: things_only without class_agnostic, or a variant that
                ``robust_filter`` did not select.
        """
        check_rf_variant(rf, self.robust_filter)
        if things_only:
            if not class_agnostic:
                raise ValueError('things_only is defined only for the class-agnostic pass '
                                 '(pass class_agnostic=True)')
            return self.stats_cath_rf if rf else self.stats_cath
        if class_agnostic:
            return self.stats_ca_rf if rf else self.stats_ca
        return self.stats_rf if rf else self.stats

    def compute_metrics(self, rf: bool = False, class_agnostic: bool = False,
                        things_only: bool = False) -> dict:
        """PQ/SQ/RQ per class and total/things/stuff of one variant.

        The flags select the stats variant as in ``raw_stats``. For the
        class-agnostic variants ``total`` holds the single class-agnostic
        number (things/stuff splits are meaningless there).
        """
        return compute_pq_from_stats(self.raw_stats(rf, class_agnostic, things_only), self.label)
