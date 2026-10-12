"""VC-PQ (View-Consistent Panoptic Quality) metric accumulator.

VC-PQ evaluates multi-view panoptic segmentation by jointly measuring:
1. Per-view segmentation quality (I: mean IoU over co-visible views)
2. Multi-view consistency (C: support overlap ratio)
3. Scene-level instance ID consistency (via one-to-one matching)

Unlike PQ-Scene (which stacks views and lets large masks dominate),
VC-PQ gives equal weight to each visible view (γ=0).

pair score (γ-family):
    q_γ(g, p) = Σ_{v ∈ S(g) ∪ S(p)} w_v^(γ) * IoU_v(g, p)
    w_v^(γ) = u_v^γ / Σ u_u^γ   (u_v = |M_v(g) ∪ M_v(p)|)

    γ=0: equal-view weighting (recommended, default)
    γ=1: area-weighted (equivalent to concat IoU)

class-level:
    VC-PQ_c = Σ q / (N_TP + 0.5*N_FP + 0.5*N_FN)

dataset-level:
    VC-PQ = (1/|C+|) * Σ_{c ∈ C+} VC-PQ_c

Void convention (same as COCO PQ):
    Predicted pixels falling on GT void (gt_pan == 0) are treated as if not
    predicted at all — removed from per-view pred areas, and therefore from
    the support judgment (eta) and every IoU union. Additionally, unmatched
    predictions whose raw area lies mostly (> void_threshold) on void are
    excluded from FP counting (tracked separately as ``fp_void_skipped``).
    With a void-free GT this is bit-identical to the definition without it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

import numpy as np

from .label_systems import LabelSystem
from .overlap import panoptic_overlap_table
from .panoptic import SEM_MULT, segments_from_pan
from .robust_filter import RF_PASSES, apply_robust_filter, check_rf_variant


# Sentinel class id for the class-agnostic pass: every segment is relabelled to
# this single dummy class so matching ignores the class head entirely.
_CA_CLASS = -1

# Loss-accounting keys. Every FN / FP is attributed to one cause
# from the two factors of q = IoU-bar * VC and the matching threshold tau:
#   a  no same-class candidate with IoU-bar > tau        (mask / detection)
#   b  such a candidate exists but none has VC > tau      (view inconsistency)
#   c  a candidate has both > tau, only the product fails (threshold)
# TP shortfalls are derived in _compute from tp / i_sum / q_sum:
#   S_mask = tp - i_sum = sum(1 - IoU-bar),  S_vc = i_sum - q_sum = sum(IoU-bar (1 - VC)).
_PARTITION_KEYS = ('fn_a', 'fn_b', 'fn_c', 'fp_a', 'fp_b', 'fp_c')


def _new_stat() -> dict:
    """Fresh per-class stat record (shared by all six stat variants)."""
    s = dict(tp=0, fp=0, fn=0, q_sum=0.0, i_sum=0.0, c_sum=0.0, fp_void_skipped=0)
    s.update({k: 0 for k in _PARTITION_KEYS})
    return s


def _partition_kind(i_row: np.ndarray, c_row: np.ndarray, tau: float) -> str:
    """Attribute one unmatched instance to 'a' / 'b' / 'c' (see _PARTITION_KEYS).

    Args:
        i_row: IoU-bar against every same-class counterpart, shape [n].
        c_row: VC against the same counterparts, shape [n].
        tau: matching threshold.
    """
    cand = i_row > tau
    if not cand.any():
        return 'a'
    if (c_row[cand] > tau).any():
        return 'c'
    return 'b'


class VCPQAccumulator:
    """Scene-level VC-PQ accumulator.

    Collects per-frame panoptic maps, then at scene end computes
    visibility-clipped masks, support sets, pair scores, one-to-one
    matching, and TP/FP/FN.

    Usage::

        accum = VCPQAccumulator(label=load_label_system('spp100'))
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
    ``class_agnostic`` / ``things_only`` flags. VC-PQ itself is robust_filter_off.

    Args:
        label: Label system; its ``thing_ids`` / ``stuff_ids`` define the
            things/stuff averages and the things-only class-agnostic pass.
        eta: Visibility threshold — minimum pixel count for a mask to be
            considered visible in a view. Default: 100.
        tau: Matching threshold. Pairs with q(g,p) <= tau are excluded
            from TP candidates. Default: 0.5 (strict inequality).
        gamma: View-weighting exponent. 0=equal-view (recommended),
            1=area-weighted (concat IoU). Default: 0.0.
        void_threshold: Fraction of a prediction's raw area on GT void above
            which an unmatched prediction is excluded from FP counting
            (same rule as PQ). Default: 0.5.
        robust_filter_threshold: Robust-class filter threshold of the robust_filter_on variant: a
            class occupying < robust_filter_threshold of the non-void pixels of the
            whole chunk, in pred OR GT, is voided before matching.
            Default: 0.005 (Panoptic Lifting). 0 disables the filter
            (robust_filter_on == robust_filter_off).
        class_agnostic: Also run the class-agnostic passes (every segment
            relabelled to one dummy class). Default: False.
        robust_filter: Which variants to accumulate: ``'off'`` (robust_filter_off
            only, VC-PQ), ``'on'`` (robust_filter_on only) or ``'both'``. Default:
            ``'both'``.

    Attributes:
        scene_cache: Optional dict shared by several accumulators that see
            the SAME frames and differ only in gamma (offline gamma sweeps);
            set it before ``end_scene``. None (default) = no sharing.
    """

    def __init__(
        self,
        label: LabelSystem,
        eta: int = 100,
        tau: float = 0.5,
        gamma: float = 0.0,
        void_threshold: float = 0.5,
        robust_filter_threshold: float = 0.005,
        class_agnostic: bool = False,
        robust_filter: str = 'both',
    ) -> None:
        if robust_filter not in RF_PASSES:
            raise ValueError(f"robust_filter must be 'off', 'on' or 'both', got {robust_filter!r}")
        self.label = label
        self.eta = eta
        self.tau = tau
        self.gamma = gamma
        self.void_threshold = void_threshold
        # Panoptic-Lifting robust-class filter of the robust_filter_on variant. A class
        # occupying < robust_filter_threshold of the (non-void) pixels of the whole chunk
        # (all views concatenated), in pred OR GT, is voided before matching.
        # Same rule as PQSceneAccumulator; VC-PQ's own eta gate is orthogonal.
        self.robust_filter_threshold = robust_filter_threshold
        self.thing_ids = label.thing_ids
        self.stuff_ids = label.stuff_ids
        # Class-agnostic pass (off by default): when on, end_scene runs two extra
        # matching passes with every segment relabelled to _CA_CLASS, so the class
        # head is removed from matching. robust_filter_off (original maps) -> stats_ca,
        # robust_filter_on (robust-filtered maps) -> stats_ca_rf.
        self.class_agnostic = class_agnostic
        # Which of the robust_filter_off / robust_filter_on passes end_scene runs.
        self.robust_filter = robust_filter
        self._run_rf_off, self._run_rf_on = RF_PASSES[robust_filter]
        # Optional per-scene cache shared by several accumulators that see the SAME
        # frames but differ only in gamma (offline gamma sweeps). gamma enters only
        # the pair-score weighting (step 4), so the robust-filtered maps, the
        # things-only maps and the step-2 overlap tables are identical across them
        # and computed once. None (default) = no sharing, same code path.
        self.scene_cache: Optional[dict] = None
        self._rf_phase = False   # True while end_scene runs the robust_filter_on passes

        # robust_filter_off (self.stats, unfiltered) and robust_filter_on (self.stats_rf, robust-filtered),
        # each filled when robust_filter selects it.
        self.stats: Dict[int, dict] = defaultdict(
            _new_stat
        )
        self.stats_rf: Dict[int, dict] = defaultdict(
            _new_stat
        )
        # Class-agnostic stats (populated only when self.class_agnostic).
        self.stats_ca: Dict[int, dict] = defaultdict(
            _new_stat
        )
        self.stats_ca_rf: Dict[int, dict] = defaultdict(
            _new_stat
        )
        # Things-only class-agnostic: stuff voided in pred AND GT maps and
        # dropped from the segment lists, remaining things collapsed to _CA_CLASS.
        # robust_filter_off -> stats_cath, robust_filter_on -> stats_cath_rf.
        self.stats_cath: Dict[int, dict] = defaultdict(
            _new_stat
        )
        self.stats_cath_rf: Dict[int, dict] = defaultdict(
            _new_stat
        )

        self._reset_scene()

    def _reset_scene(self):
        self._pred_pans: list[np.ndarray] = []
        self._gt_pans: list[np.ndarray] = []
        self._pred_segments_list: list[list[dict]] = []
        self._gt_segments_list: list[list[dict]] = []

    def begin_scene(self) -> None:
        """Call before adding frames of a new scene / chunk."""
        self._reset_scene()

    def add_frame(self, pred_pan: np.ndarray, gt_pan: np.ndarray) -> None:
        """Add one frame's panoptic maps to the current scene.

        Segment lists are derived from the maps (``segments_from_pan``).

        Args:
            pred_pan: [H, W] predicted panoptic map (sem*1000 + inst, 0 = no
                prediction; cross-view consistent IDs).
            gt_pan: [H, W] GT panoptic map (scene-level consistent IDs,
                0 = void).
        """
        self._pred_pans.append(pred_pan)
        self._gt_pans.append(gt_pan)
        self._pred_segments_list.append(segments_from_pan(pred_pan, self.thing_ids))
        self._gt_segments_list.append(segments_from_pan(gt_pan, self.thing_ids))

    def _apply_robust_filter(self) -> None:
        """Void non-robust classes across the chunk (Panoptic-Lifting faithful).

        A semantic class occupying < ``robust_filter_threshold`` of the non-void
        (gt != 0) pixels of the whole chunk (all views concatenated), in the
        prediction OR the GT, is set to void (0) in every view's pred and GT
        map. Voided instances then fall below the eta visibility gate and are
        dropped; predicted pixels landing on the newly-voided GT are handled by
        the existing void convention. Mirrors PQSceneAccumulator's filter.
        """
        thr = self.robust_filter_threshold
        cache = self.scene_cache
        ckey = ('rf_maps', thr)
        if cache is not None and ckey in cache:
            self._pred_pans, self._gt_pans = cache[ckey]   # read-only downstream
            return
        self._pred_pans, self._gt_pans = apply_robust_filter(self._pred_pans, self._gt_pans, thr)
        if cache is not None:
            cache[ckey] = (self._pred_pans, self._gt_pans)

    def end_scene(self) -> None:
        """Accumulate robust_filter_off (self.stats) and / or robust_filter_on (self.stats_rf).

        The matching pipeline runs on the original maps (robust_filter_off), then on the
        robust-filtered maps with self.stats temporarily pointed at stats_rf
        (robust_filter_on), each when ``robust_filter`` selects it. When class_agnostic is
        on, each selected variant gets two more passes (class collapsed to
        _CA_CLASS): stats_ca / stats_cath (original maps) and stats_ca_rf /
        stats_cath_rf (robust-filtered maps).
        """
        if not self._pred_pans:
            self._reset_scene()
            return
        if self._run_rf_off:
            # robust_filter_off: original maps -> self.stats
            self._pipeline()
            # class-agnostic robust_filter_off: original maps, class collapsed -> stats_ca
            if self.class_agnostic:
                self._run_pipeline_into(self.stats_ca, collapse=True)
                self._run_pipeline_into(self.stats_cath, collapse=True, things_only=True)
        if not self._run_rf_on:
            self._reset_scene()
            return
        # robust_filter_on: robust-filtered maps -> self.stats_rf (via a stats swap)
        self._apply_robust_filter()   # rebinds self._pred_pans / _gt_pans to filtered copies
        self._rf_phase = True
        _save_stats = self.stats
        self.stats = self.stats_rf
        try:
            self._pipeline()
        finally:
            self.stats = _save_stats
        # class-agnostic robust_filter_on: robust-filtered maps, class collapsed -> stats_ca_rf
        if self.class_agnostic:
            self._run_pipeline_into(self.stats_ca_rf, collapse=True)
            self._run_pipeline_into(self.stats_cath_rf, collapse=True, things_only=True)
        self._rf_phase = False
        self._reset_scene()

    def _run_pipeline_into(
        self, stats_dict: Dict[int, dict], collapse: bool, things_only: bool = False,
    ) -> None:
        """Run one matching pass over the current maps, accumulating into stats_dict.

        Temporarily repoints self.stats so the shared pipeline writes elsewhere;
        used for the class-agnostic passes (see end_scene). With ``things_only``
        the pass sees copies of the maps in which every stuff pixel is void and
        segment lists without stuff entries (instance-segmentation convention).
        """
        _save = (self.stats, self._pred_pans, self._gt_pans,
                 self._pred_segments_list, self._gt_segments_list)
        self.stats = stats_dict
        try:
            if things_only:
                cache = self.scene_cache
                ckey = ('things_only', self._rf_phase)
                if cache is not None and ckey in cache:
                    (self._pred_pans, self._gt_pans,
                     self._pred_segments_list, self._gt_segments_list) = cache[ckey]
                else:
                    th = list(self.thing_ids)

                    def _void_stuff(pan: np.ndarray) -> np.ndarray:
                        out = pan.copy()
                        sem = out // SEM_MULT
                        out[(sem != 0) & ~np.isin(sem, th)] = 0
                        return out

                    self._pred_pans = [_void_stuff(p) for p in self._pred_pans]
                    self._gt_pans = [_void_stuff(g) for g in self._gt_pans]
                    self._pred_segments_list = [[s for s in segs if s['category_id'] in self.thing_ids]
                                                for segs in self._pred_segments_list]
                    self._gt_segments_list = [[s for s in segs if s['category_id'] in self.thing_ids]
                                              for segs in self._gt_segments_list]
                    if cache is not None:
                        cache[ckey] = (self._pred_pans, self._gt_pans,
                                       self._pred_segments_list, self._gt_segments_list)
            self._pipeline(collapse=collapse, things_only=things_only)
        finally:
            (self.stats, self._pred_pans, self._gt_pans,
             self._pred_segments_list, self._gt_segments_list) = _save

    def _pipeline(self, collapse: bool = False, things_only: bool = False) -> None:
        """VC-PQ matching over the currently collected maps, accumulating self.stats.

        Steps:
        1. Extract per-instance per-view areas and pairwise intersections
        2. Apply visibility threshold (eta) → support sets
        3. Compute pair scores q(g,p) = I * C for same-class pairs
        4. Maximum-weight bipartite matching (Hungarian, q > tau)
        5. Accumulate TP/FP/FN into self.stats

        ``things_only`` only names the map variant for the shared scene cache.
        """
        n_views = len(self._pred_pans)

        # ── Step 1: Collect instance IDs and classes ──
        gt_instances: Dict[int, int] = {}   # pan_id → category_id
        pred_instances: Dict[int, int] = {}

        for frame_segs in self._gt_segments_list:
            for seg in frame_segs:
                gt_instances[seg['id']] = seg['category_id']
        for frame_segs in self._pred_segments_list:
            for seg in frame_segs:
                pred_instances[seg['id']] = seg['category_id']

        if collapse:
            # Class-agnostic: relabel every instance to one dummy class so the
            # same-class filter below admits all pairs and Hungarian matching
            # ignores the class head entirely.
            gt_instances = {gid: _CA_CLASS for gid in gt_instances}
            pred_instances = {pid: _CA_CLASS for pid in pred_instances}

        # ── Step 2: Per-instance per-view areas + pairwise intersections ──
        # Using combined-ID trick for O(H*W) intersection per view.
        # gamma-independent (only eta, the map variant and the class relabel enter),
        # so with a shared scene cache it is computed once per variant.
        cache = self.scene_cache
        ckey = ('step2', self.eta, self._rf_phase, collapse, things_only)
        if cache is not None and ckey in cache:
            gt_areas, pred_areas, pair_inter, pred_void_frac = cache[ckey]   # read-only
        else:
            gt_areas, pred_areas, pair_inter, pred_void_frac = self._step2_overlaps(
                n_views, gt_instances, pred_instances)
            if cache is not None:
                cache[ckey] = (gt_areas, pred_areas, pair_inter, pred_void_frac)

        self._step3_to_5(gt_instances, pred_instances, gt_areas, pred_areas,
                         pair_inter, pred_void_frac)

    def _step2_overlaps(self, n_views: int, gt_instances: Dict[int, int],
                        pred_instances: Dict[int, int]) -> tuple:
        """Step 2: per-instance per-view areas, same-class pairwise intersections,
        and per-prediction void fraction (all gamma-independent)."""
        gt_areas: Dict[int, Dict[int, int]] = defaultdict(dict)    # gid → {v: area}
        pred_areas: Dict[int, Dict[int, int]] = defaultdict(dict)  # pid → {v: void-free area}
        pair_inter: Dict[tuple, Dict[int, int]] = defaultdict(dict)  # (gid,pid) → {v: count}
        pred_raw_total: Dict[int, int] = defaultdict(int)   # pid → raw px, all views
        pred_void_total: Dict[int, int] = defaultdict(int)  # pid → void px, all views

        for v in range(n_views):
            # Areas and pairwise intersections in one pass (see
            # vcpq/overlap.py). Instances present in a map but
            # absent from that view's segment list were computed before and
            # then dropped by the same-class filter below, so restricting the
            # table to the listed ids changes nothing.
            g_ids_v = [s['id'] for s in self._gt_segments_list[v]]
            p_ids_v = [s['id'] for s in self._pred_segments_list[v]]
            table = panoptic_overlap_table(
                self._pred_pans[v], self._gt_pans[v], p_ids_v, g_ids_v)

            # Per-instance areas (with visibility clipping)
            for gi, gid in enumerate(g_ids_v):
                area = int(table.gt_area[gi])
                if area >= self.eta:
                    gt_areas[gid][v] = area

            for pi, pid in enumerate(p_ids_v):
                raw = int(table.pred_area[pi])
                void_px = int(table.pred_void[pi])
                pred_raw_total[pid] += raw
                pred_void_total[pid] += void_px
                # Void convention (module docstring): predicted pixels on GT void
                # are removed before any scoring — this area feeds the support
                # judgment (eta) and the per-view IoU unions below.
                area = raw - void_px
                if area >= self.eta:
                    pred_areas[pid][v] = area

            for gi, gid in enumerate(g_ids_v):
                # Only record if both pass visibility threshold in this view
                if v not in gt_areas.get(gid, {}):
                    continue
                # Same-class filter (skip cross-class overlaps)
                g_cat = gt_instances.get(gid)
                if g_cat is None:
                    continue
                row = table.inter[gi]
                for pi in np.nonzero(row)[0]:
                    pid = p_ids_v[pi]
                    if v not in pred_areas.get(pid, {}):
                        continue
                    if pred_instances.get(pid) != g_cat:
                        continue
                    pair_inter[(gid, pid)][v] = int(row[pi])

        pred_void_frac: Dict[int, float] = {
            pid: (pred_void_total[pid] / pred_raw_total[pid])
            if pred_raw_total[pid] > 0 else 0.0
            for pid in pred_raw_total
        }
        return gt_areas, pred_areas, pair_inter, pred_void_frac

    def _step3_to_5(self, gt_instances: Dict[int, int],
                    pred_instances: Dict[int, int], gt_areas: dict, pred_areas: dict,
                    pair_inter: dict, pred_void_frac: Dict[int, float]) -> None:
        """Steps 3-5: support sets, gamma-weighted pair scores, matching, accumulation."""
        # ── Step 3: Build support sets and group by class ──
        gt_support: Dict[int, set] = {
            gid: set(views.keys()) for gid, views in gt_areas.items()
        }
        pred_support: Dict[int, set] = {
            pid: set(views.keys()) for pid, views in pred_areas.items()
        }

        gt_by_class: Dict[int, list] = defaultdict(list)
        pred_by_class: Dict[int, list] = defaultdict(list)

        for gid, cat in gt_instances.items():
            if gt_support.get(gid):
                gt_by_class[cat].append(gid)
        for pid, cat in pred_instances.items():
            if pred_support.get(pid):
                pred_by_class[cat].append(pid)

        all_classes = set(gt_by_class.keys()) | set(pred_by_class.keys())

        # ── Step 4 & 5: Per-class pair scores → matching → accumulate ──
        for cat in all_classes:
            g_ids = gt_by_class.get(cat, [])
            p_ids = pred_by_class.get(cat, [])
            n_gt = len(g_ids)
            n_pred = len(p_ids)

            if n_gt == 0 and n_pred == 0:
                continue
            if n_gt == 0:
                self._count_fps(cat, p_ids, pred_void_frac)
                continue
            if n_pred == 0:
                self.stats[cat]['fn'] += n_gt
                self.stats[cat]['fn_a'] += n_gt      # no candidate at all
                continue

            # Compute q, I, C matrices
            q_mat = np.zeros((n_gt, n_pred), dtype=np.float64)
            i_mat = np.zeros((n_gt, n_pred), dtype=np.float64)
            c_mat = np.zeros((n_gt, n_pred), dtype=np.float64)

            for gi, gid in enumerate(g_ids):
                g_sup = gt_support[gid]
                for pi, pid in enumerate(p_ids):
                    p_sup = pred_support[pid]
                    union_sup = g_sup | p_sup
                    inter_sup = g_sup & p_sup

                    if not union_sup:
                        continue

                    # Per-view IoU and γ-weighted pair score
                    iou_sum = 0.0
                    weighted_iou = 0.0
                    weight_sum = 0.0
                    inter_views = pair_inter.get((gid, pid), {})

                    for v in union_sup:
                        if v in inter_sup:
                            inter_px = inter_views.get(v, 0)
                            union_px = gt_areas[gid][v] + pred_areas[pid][v] - inter_px
                            iou_v = inter_px / union_px if union_px > 0 else 0.0
                            u_v = max(union_px, 1)
                        elif v in g_sup:
                            iou_v = 0.0
                            u_v = gt_areas[gid][v]
                        else:
                            iou_v = 0.0
                            u_v = pred_areas[pid][v]

                        iou_sum += iou_v
                        w = u_v ** self.gamma
                        weight_sum += w
                        weighted_iou += w * iou_v

                    n_union = len(union_sup)
                    n_inter = len(inter_sup)

                    q_mat[gi, pi] = weighted_iou / weight_sum if weight_sum > 0 else 0.0
                    # I, C always use equal-view definition (analysis metrics)
                    i_mat[gi, pi] = iou_sum / n_inter if n_inter > 0 else 0.0
                    c_mat[gi, pi] = n_inter / n_union

            # Hungarian matching on valid pairs (q > tau)
            self._match_and_accumulate(
                cat, q_mat, i_mat, c_mat, n_gt, n_pred, p_ids, pred_void_frac)

    def _count_fps(
        self,
        cat: int,
        pids: list,
        pred_void_frac: Optional[Dict[int, float]],
        fp_kind: Optional[Dict[int, str]] = None,
    ) -> None:
        """Count unmatched predictions as FP, minus the void-majority ones.

        Mirrors COCO PQ / PQ-Scene: an unmatched prediction whose raw
        area lies mostly (> void_threshold) on GT void is not a scoring
        failure — it is dropped from the FP count and tracked separately in
        ``fp_void_skipped`` (void convention, module docstring).

        Args:
            fp_kind: pid -> 'a' | 'b' | 'c' loss-accounting cause; absent
                pids (e.g. a class with no GT) default to 'a'. Void-skipped
                predictions enter no partition bucket, so
                fp == fp_a + fp_b + fp_c always holds.
        """
        for pid in pids:
            vf = pred_void_frac.get(pid, 0.0) if pred_void_frac else 0.0
            skipped = vf > self.void_threshold
            if skipped:
                self.stats[cat]['fp_void_skipped'] += 1
            else:
                self.stats[cat]['fp'] += 1
                kind = fp_kind.get(pid, 'a') if fp_kind else 'a'
                self.stats[cat]['fp_' + kind] += 1

    def _match_and_accumulate(
        self,
        cat: int,
        q_mat: np.ndarray,
        i_mat: np.ndarray,
        c_mat: np.ndarray,
        n_gt: int,
        n_pred: int,
        p_ids: list,
        pred_void_frac: Dict[int, float],
    ) -> None:
        """Run Hungarian matching and accumulate TP/FP/FN for one class.

        Args:
            cat: Semantic class ID.
            q_mat: [n_gt, n_pred] pair score matrix.
            i_mat: [n_gt, n_pred] segmentation term matrix.
            c_mat: [n_gt, n_pred] consistency term matrix.
            n_gt: Number of GT instances.
            n_pred: Number of predicted instances.
            p_ids: Predicted instance ids, in the column order of the matrices.
            pred_void_frac: pid -> fraction of its raw area on GT void.
        """
        from scipy.optimize import linear_sum_assignment

        valid = q_mat > self.tau

        matched_gt: set = set()
        matched_pred: set = set()

        if valid.any():
            # Maximum-weight matching: minimize negative cost.
            # Prohibit invalid edges with large cost.
            cost = -q_mat.copy()
            cost[~valid] = 1e6
            row_idx, col_idx = linear_sum_assignment(cost)
        else:
            row_idx, col_idx = np.array([], dtype=int), np.array([], dtype=int)

        for r, c in zip(row_idx, col_idx):
            if q_mat[r, c] > self.tau:
                matched_gt.add(r)
                matched_pred.add(c)
                self.stats[cat]['tp'] += 1
                self.stats[cat]['q_sum'] += q_mat[r, c]
                self.stats[cat]['i_sum'] += i_mat[r, c]
                self.stats[cat]['c_sum'] += c_mat[r, c]

        # Loss accounting: attribute every unmatched pred / GT to a / b / c
        # (see _PARTITION_KEYS) from its IoU-bar and VC rows.
        unmatched_cols = [c for c in range(n_pred) if c not in matched_pred]
        col_kind = {c: _partition_kind(i_mat[:, c], c_mat[:, c], self.tau)
                    for c in unmatched_cols}
        unmatched_pids = [p_ids[c] for c in unmatched_cols]
        fp_kind = {p_ids[c]: k for c, k in col_kind.items()}
        self._count_fps(cat, unmatched_pids, pred_void_frac, fp_kind)
        for r in range(n_gt):
            if r in matched_gt:
                continue
            self.stats[cat]['fn'] += 1
            kind = _partition_kind(i_mat[r, :], c_mat[r, :], self.tau)
            self.stats[cat]['fn_' + kind] += 1

    def merge(self, other: 'VCPQAccumulator') -> None:
        """Merge another accumulator's stats into this one (all variants)."""
        for dst, src in ((self.stats, other.stats), (self.stats_rf, other.stats_rf),
                         (self.stats_ca, other.stats_ca),
                         (self.stats_ca_rf, other.stats_ca_rf),
                         (self.stats_cath, other.stats_cath),
                         (self.stats_cath_rf, other.stats_cath_rf)):
            for cat, s in src.items():
                dst[cat]['tp'] += s['tp']
                dst[cat]['fp'] += s['fp']
                dst[cat]['fn'] += s['fn']
                dst[cat]['q_sum'] += s['q_sum']
                dst[cat]['i_sum'] += s['i_sum']
                dst[cat]['c_sum'] += s['c_sum']
                dst[cat]['fp_void_skipped'] += s.get('fp_void_skipped', 0)
                for k in _PARTITION_KEYS:
                    dst[cat][k] += s.get(k, 0)

    def raw_stats(self, rf: bool = False, class_agnostic: bool = False,
                  things_only: bool = False) -> Dict[int, dict]:
        """The accumulated per-class stats dict of one variant (not a copy).

        {cat: {tp, fp, fn, q_sum, i_sum, c_sum, fp_void_skipped,
               fn_a, fn_b, fn_c, fp_a, fp_b, fp_c}}

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
        """VC-PQ/SQ/RQ per class and total/things/stuff of one variant.

        The flags select the stats variant as in ``raw_stats``. For the
        class-agnostic variants ``total`` holds the single class-agnostic
        number (things/stuff splits are meaningless there).
        """
        return self._compute(self.raw_stats(rf, class_agnostic, things_only))

    def _compute(self, stats: Dict[int, dict]) -> dict:
        """Compute VC-PQ / VC-SQ / VC-RQ per class and total/things/stuff.

        Class averaging follows panopticapi: a class with tp + fp + fn == 0
        (present only as void-majority predictions) is left out of every
        mean instead of entering as 0.0. Such classes are counted in
        ``n_void_only``. PQ-Scene keeps the Panoptic Lifting rule and is
        unaffected.

        Returns:
            Dict with keys 'per_class', 'total', 'things', 'stuff'.
            per_class entries include mean_I / mean_C and the loss accounting
            (fn_a..fp_c, s_mask, s_vc, acct_*); for denom > 0
            vcpq + acct_vc + acct_mask + acct_thr == 1.
        """
        per_class: dict = {}
        for cat in sorted(stats.keys()):
            s = stats[cat]
            tp, fp, fn = s['tp'], s['fp'], s['fn']
            denom = tp + 0.5 * fp + 0.5 * fn

            vcpq = s['q_sum'] / denom if denom > 0 else 0.0
            vcsq = s['q_sum'] / tp if tp > 0 else 0.0
            vcrq = tp / denom if denom > 0 else 0.0
            mean_i = s['i_sum'] / tp if tp > 0 else 0.0
            mean_c = s['c_sum'] / tp if tp > 0 else 0.0

            # TP shortfall split: 1 - q = (1 - IoU-bar) + IoU-bar (1 - VC).
            s_mask = tp - s['i_sum']
            s_vc = s['i_sum'] - s['q_sum']
            part = {k: s.get(k, 0) for k in _PARTITION_KEYS}
            if denom > 0:
                acct_vc = (s_vc + 0.5 * part['fn_b'] + 0.5 * part['fp_b']) / denom
                acct_mask = (s_mask + 0.5 * part['fn_a'] + 0.5 * part['fp_a']) / denom
                acct_thr = (0.5 * part['fn_c'] + 0.5 * part['fp_c']) / denom
            else:
                acct_vc = acct_mask = acct_thr = 0.0

            per_class[cat] = dict(
                vcpq=vcpq, vcsq=vcsq, vcrq=vcrq,
                mean_i=mean_i, mean_c=mean_c,
                tp=tp, fp=fp, fn=fn,
                q_sum=s['q_sum'], i_sum=s['i_sum'], c_sum=s['c_sum'],
                fp_void_skipped=s['fp_void_skipped'],
                s_mask=s_mask, s_vc=s_vc,
                acct_vc=acct_vc, acct_mask=acct_mask, acct_thr=acct_thr,
                **part,
            )

        _MEAN_KEYS = ('vcpq', 'vcsq', 'vcrq', 'mean_i', 'mean_c',
                      'acct_vc', 'acct_mask', 'acct_thr')

        def _avg(ids: set) -> dict:
            present = [per_class[c] for c in ids if c in per_class]
            vals = [v for v in present if v['tp'] + v['fp'] + v['fn'] > 0]
            out = {k: 0.0 for k in _MEAN_KEYS}
            out['n_classes'] = len(vals)
            out['n_void_only'] = len(present) - len(vals)
            if vals:
                out.update({k: float(np.mean([v[k] for v in vals])) for k in _MEAN_KEYS})
            return out

        all_ids = set(per_class.keys())
        return dict(
            per_class=per_class,
            total=_avg(all_ids),
            things=_avg(all_ids & self.thing_ids),
            stuff=_avg(all_ids & self.stuff_ids),
        )
