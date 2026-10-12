"""Pixel-overlap table shared by every panoptic metric accumulator.

PQ-Scene and VC-PQ both need the same three quantities for one (pred, gt)
panoptic map pair: each segment's area, each
predicted segment's overlap with GT void, and the pairwise intersection of
every (gt, pred) segment pair.

The straightforward implementation materialises ``pred_pan == pid`` and
``gt_pan == gid`` inside a nested loop, which costs ``n_pred * n_gt`` passes
over the whole image (and recomputes every GT mask once per prediction). This
module computes all of it in three passes total, independent of the segment
counts: map both id images onto small dense indices, combine them into one
code image, and histogram the codes. The resulting contingency matrix is a few
thousand entries, so every downstream quantity is a slice of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def panoptic_ids(pan_map: np.ndarray) -> np.ndarray:
    """Sorted non-zero segment IDs of a panoptic map.

    Same result as ``np.unique(pan_map)[1:]`` but histogram-based: panoptic IDs
    are ``sem_id * 1000 + inst_id`` and therefore bounded, so counting beats
    sorting several million pixels (which is what PQ-Scene does on its stacked
    multi-view map, twice per chunk).
    """
    if pan_map.size == 0:
        return np.zeros(0, dtype=np.int64)
    counts = np.bincount(pan_map.ravel())
    ids = np.nonzero(counts)[0]
    return ids[1:] if ids.size and ids[0] == 0 else ids


@dataclass
class OverlapTable:
    """Contingency table of one (pred, gt) panoptic map pair.

    Rows/columns follow the order of the ``gt_ids`` / ``pred_ids`` lists given
    to :func:`panoptic_overlap_table`, so ``inter[gi, pi]`` pairs with
    ``gt_ids[gi]`` and ``pred_ids[pi]``.

    Attributes:
        inter: [n_gt, n_pred] pixels shared by GT segment gi and pred segment pi.
        gt_area: [n_gt] pixel count of each GT segment.
        pred_area: [n_pred] pixel count of each predicted segment.
        pred_void: [n_pred] pixels of each predicted segment falling on GT void
            (``gt_pan == 0``).
    """

    inter: np.ndarray
    gt_area: np.ndarray
    pred_area: np.ndarray
    pred_void: np.ndarray


def panoptic_overlap_table(
    pred_pan: np.ndarray,
    gt_pan: np.ndarray,
    pred_ids: list[int],
    gt_ids: list[int],
) -> OverlapTable:
    """Build the (gt x pred) overlap table in three passes over the maps.

    Args:
        pred_pan: [H, W] predicted panoptic map (sem_id * 1000 + inst_id).
        gt_pan: [H, W] GT panoptic map, 0 = void.
        pred_ids: Predicted segment IDs, in the caller's order.
        gt_ids: GT segment IDs, in the caller's order.

    Returns:
        OverlapTable aligned with ``gt_ids`` / ``pred_ids``.

    Note:
        IDs present in a map but absent from the id lists are pooled into a
        spare row/column and never reported, so a partial id list (e.g. things
        only) still yields exact areas for the ids it does list. GT id 0 keeps
        its own row, which is what makes ``pred_void`` exact.
    """
    n_gt, n_pred = len(gt_ids), len(pred_ids)
    # Row 0 = GT void, rows 1..n_gt = gt_ids, row n_gt+1 = any other GT id.
    # Column 0 = pred background, columns 1..n_pred = pred_ids, last = other.
    n_rows, n_cols = n_gt + 2, n_pred + 2

    # The lists may name a segment that no longer exists in the map (e.g. after
    # class exclusion), so size the lookup tables for both sources.
    max_id = int(max(pred_pan.max(initial=0), gt_pan.max(initial=0),
                     max(pred_ids, default=0), max(gt_ids, default=0)))
    gt_lut = np.full(max_id + 1, n_gt + 1, dtype=np.int32)
    pred_lut = np.full(max_id + 1, n_pred + 1, dtype=np.int32)
    gt_lut[0] = 0
    pred_lut[0] = 0
    if n_gt:
        gt_lut[np.asarray(gt_ids, dtype=np.int64)] = np.arange(1, n_gt + 1, dtype=np.int32)
    if n_pred:
        pred_lut[np.asarray(pred_ids, dtype=np.int64)] = np.arange(1, n_pred + 1, dtype=np.int32)

    codes = gt_lut[gt_pan] * np.int32(n_cols) + pred_lut[pred_pan]
    counts = np.bincount(codes.ravel(), minlength=n_rows * n_cols)
    counts = counts.reshape(n_rows, n_cols)

    return OverlapTable(
        inter=counts[1:n_gt + 1, 1:n_pred + 1],
        gt_area=counts[1:n_gt + 1, :].sum(axis=1),
        pred_area=counts[:, 1:n_pred + 1].sum(axis=0),
        pred_void=counts[0, 1:n_pred + 1],
    )
