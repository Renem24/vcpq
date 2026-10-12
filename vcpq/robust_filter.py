"""Panoptic-Lifting robust-class filter (single implementation).

A semantic class occupying less than ``thr`` of the non-void (``gt != 0``)
pixels of a chunk (all views pooled) — in the PREDICTION or the GT — is
considered too rare to be robustly evaluated. Its pixels are voided (set to 0)
in both maps, in every view, so it contributes no TP/FP/FN downstream.

Only classes that actually occur (nonzero pixel count) in a given side's pooled histogram are checked against
that side's fraction — a class entirely absent from one side contributes no
check from that side (it is not synthesised as a 0/total fraction there), but
it can still be voided via the OTHER side's fraction.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .panoptic import SEM_MULT

# ``robust_filter`` option of the accumulators -> (accumulate the robust_filter_off
# variant, accumulate the robust_filter_on variant).
RF_PASSES = {'off': (True, False), 'on': (False, True), 'both': (True, True)}


def check_rf_variant(rf: bool, robust_filter: str) -> None:
    """ValueError if an accumulator built with ``robust_filter`` did not
    accumulate the robust_filter_on (``rf``) or robust_filter_off variant."""
    if not RF_PASSES[robust_filter][1 if rf else 0]:
        raise ValueError(f"the robust_filter_{'on' if rf else 'off'} variant was not computed "
                         f"(robust_filter={robust_filter!r})")


def non_robust_classes(
    pred_pans: list[np.ndarray], gt_pans: list[np.ndarray], thr: float = 0.005
) -> set[int]:
    """Semantic classes occupying < ``thr`` of the chunk's non-void GT pixels.

    Args:
        pred_pans: Per-view predicted panoptic maps (sem * SEM_MULT + inst).
        gt_pans: Per-view GT panoptic maps, 0 = void.
        thr: Robustness threshold (Panoptic-Lifting default 0.005 = 0.5%).

    Returns:
        Semantic class ids to void, pooled over all views.
    """
    total = 0
    pred_counts: dict[int, int] = defaultdict(int)
    gt_counts: dict[int, int] = defaultdict(int)
    for pred, gt in zip(pred_pans, gt_pans):
        valid = gt != 0
        total += int(valid.sum())
        for arr, counts in ((pred, pred_counts), (gt, gt_counts)):
            cnt = np.bincount((arr[valid] // SEM_MULT).ravel())
            for sid in np.nonzero(cnt)[0]:
                if sid != 0:
                    counts[int(sid)] += int(cnt[sid])
    if total == 0:
        return set()

    non_robust: set[int] = set()
    for counts in (pred_counts, gt_counts):
        for sid, cnt in counts.items():
            if cnt / total < thr:
                non_robust.add(sid)
    return non_robust


def apply_robust_filter(
    pred_pans: list[np.ndarray], gt_pans: list[np.ndarray], thr: float = 0.005
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Void every pixel of a non-robust class, in both maps of every view.

    Returns new per-view arrays; ``pred_pans``/``gt_pans`` are never modified.
    """
    non_robust = non_robust_classes(pred_pans, gt_pans, thr)
    pred_out, gt_out = [], []
    for pred, gt in zip(pred_pans, gt_pans):
        p, g = pred.copy(), gt.copy()
        if non_robust:
            for out in (p, g):
                sem = out // SEM_MULT
                lut = np.zeros(int(sem.max()) + 1, dtype=bool)
                in_range = [c for c in non_robust if c < lut.size]
                if in_range:
                    lut[in_range] = True
                    out[lut[sem]] = 0
        pred_out.append(p)
        gt_out.append(g)
    return pred_out, gt_out
