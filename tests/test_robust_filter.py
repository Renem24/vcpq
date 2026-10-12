"""Panoptic-Lifting robust-class filter: void classes rare across the chunk.

The toy case below deliberately does NOT rely on the "class fully absent from
one side" edge case, because with 200 total pixels in a 2x10x10 scene, no
present (nonzero-count) class can satisfy the DEFAULT thr=0.005 (1/200 == thr
exactly, not strictly less). Instead it uses an explicit thr=0.02 so that a
3/200 == 0.015 fraction, present identically in both pred and GT, crosses the
threshold on its own — this is unambiguous under the per-side
nonzero-histogram check of vcpq.robust_filter.
"""
import numpy as np

from vcpq.robust_filter import apply_robust_filter, non_robust_classes


def _make_views():
    """2 views, 10x10: class 7 = 3 px/view-pair (both pred and GT), class 1 =
    the remaining 197 px/scene (both pred and GT). total = 200."""
    view0 = np.full((10, 10), 1000, dtype=np.int32)
    view0[0, 0:3] = 7000
    view1 = np.full((10, 10), 1000, dtype=np.int32)
    pred_pans = [view0.copy(), view1.copy()]
    gt_pans = [view0.copy(), view1.copy()]
    return pred_pans, gt_pans


def test_rare_class_voided_common_class_stays():
    pred_pans, gt_pans = _make_views()

    non_robust = non_robust_classes(pred_pans, gt_pans, thr=0.02)
    assert non_robust == {7}

    out_pred, out_gt = apply_robust_filter(pred_pans, gt_pans, thr=0.02)

    for views in (out_pred, out_gt):
        assert set(np.unique(views[0]).tolist()) == {0, 1000}
        assert (views[0] == 0).sum() == 3          # the 3 voided class-7 pixels
        assert set(np.unique(views[1]).tolist()) == {1000}
        assert (views[1] == 1000).sum() == 100      # class 1 fully intact

    # inputs untouched
    assert (pred_pans[0] == 7000).sum() == 3
    assert (gt_pans[0] == 7000).sum() == 3


def test_no_rare_class_at_default_threshold():
    """At the default thr=0.005, 3/200 == 0.015 is not below threshold for a
    class present (nonzero count) on both sides, so nothing is voided."""
    pred_pans, gt_pans = _make_views()
    assert non_robust_classes(pred_pans, gt_pans) == set()
    out_pred, out_gt = apply_robust_filter(pred_pans, gt_pans)
    assert np.array_equal(out_pred[0], pred_pans[0])
    assert np.array_equal(out_gt[0], gt_pans[0])
