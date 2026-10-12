"""VC-PQ void convention (same as COCO PQ).

Predicted pixels on GT void are removed before any scoring (area, support,
IoU union), and unmatched predictions mostly on void (void_frac > 0.5) are
excluded from FP counting. With a void-free GT the void convention changes
nothing.

Run: .venv/bin/python -m pytest tests/test_vcpq_void.py
"""
import numpy as np

from vcpq.label_systems import LabelSystem
from vcpq.vcpq_accumulator import VCPQAccumulator

THING_IDS = {5, 7}
LABEL = LabelSystem(name='test_things_5_7', num_classes=8, thing_ids=frozenset(THING_IDS),
                    stuff_ids=frozenset(), class_names={}, sem_lut=None)


def _run_scene(frames) -> VCPQAccumulator:
    acc = VCPQAccumulator(label=LABEL, eta=5)
    acc.begin_scene()
    for pred_pan, gt_pan in frames:
        acc.add_frame(pred_pan, gt_pan)
    acc.end_scene()
    return acc


def test_void_bleed_does_not_lower_iou():
    """Pred covering the object exactly + extra void pixels scores q = 1.0."""
    gt = np.zeros((20, 20), dtype=np.int64)
    gt[0:10, 0:10] = 5001                      # object, 100 px; rest void

    pred = np.zeros((20, 20), dtype=np.int64)
    pred[0:10, 0:16] = 5001                    # 100 px object + 60 px void

    acc = _run_scene([(pred, gt)])
    s = acc.stats[5]
    assert s['tp'] == 1 and s['fp'] == 0 and s['fn'] == 0, s
    assert abs(s['q_sum'] - 1.0) < 1e-9, s['q_sum']
    assert abs(s['c_sum'] - 1.0) < 1e-9, s['c_sum']


def test_void_majority_fp_excluded():
    """Unmatched pred with void_frac > 0.5 is skipped; labeled-area FP counts."""
    gt = np.zeros((20, 20), dtype=np.int64)
    gt[0:10, 0:10] = 5001                      # object A (cat 5), 100 px
    gt[10:20, :] = 7001                        # wall (cat 7), 200 px
    # remaining [0:10, 10:20] = void, 100 px

    pred = np.zeros((20, 20), dtype=np.int64)
    pred[10:13, 0:10] = 5002                   # 30 px on wall (labeled)
    pred[0:7, 10:20] = 5002                    # 70 px on void → void_frac 0.7
    pred[13:18, 0:20] = 5003                   # 100 px on wall, void_frac 0.0

    acc = _run_scene([(pred, gt)])
    s5 = acc.stats[5]
    assert s5['fn'] == 1, s5                       # object A missed
    assert s5['fp'] == 1, s5                       # 5003 counted
    assert s5['fp_void_skipped'] == 1, s5          # 5002 skipped
    assert acc.stats[7]['fn'] == 1                 # wall missed

    per_class = acc.compute_metrics()['per_class']
    assert per_class[5]['fp_void_skipped'] == 1


def test_void_only_view_removed_from_support():
    """A view where the pred fires only on void does not join its support."""
    gt_vis = np.zeros((20, 20), dtype=np.int64)
    gt_vis[0:10, 0:10] = 5001                  # object visible; rest void
    pred_vis = np.zeros((20, 20), dtype=np.int64)
    pred_vis[0:10, 0:10] = 5001                # exact match

    gt_void = np.zeros((20, 20), dtype=np.int64)      # all void
    pred_void = np.zeros((20, 20), dtype=np.int64)
    pred_void[0:5, 0:10] = 5001                # 50 px, all on void

    acc = _run_scene([
        (pred_vis, gt_vis),
        (pred_vis, gt_vis),
        (pred_void, gt_void),
    ])
    s = acc.stats[5]
    assert s['tp'] == 1, s
    # Without the void convention: C = 2/3 → q ≈ 0.667. With it: view 2 leaves support → q = 1.
    assert abs(s['q_sum'] - 1.0) < 1e-9, s['q_sum']
    assert abs(s['c_sum'] - 1.0) < 1e-9, s['c_sum']


def test_no_void_scene_unchanged():
    """With a void-free GT, the void convention changes nothing."""
    gt = np.zeros((20, 20), dtype=np.int64)
    gt[0:10, :] = 5001
    gt[10:20, :] = 7001                        # no void anywhere
    pred = gt.copy()

    acc = _run_scene([(pred, gt)])
    for cat in (5, 7):
        s = acc.stats[cat]
        assert s['tp'] == 1 and s['fp'] == 0 and s['fn'] == 0, (cat, s)
        assert s['fp_void_skipped'] == 0, (cat, s)
        assert abs(s['q_sum'] - 1.0) < 1e-9, (cat, s['q_sum'])
