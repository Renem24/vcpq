"""VC-PQ loss accounting + panopticapi class-averaging.

Two behaviors:

1. Class averaging follows panopticapi: a class with tp + fp + fn == 0
   (only void-majority predictions, i.e. "void-only") is left out of the
   VC-PQ mean instead of entering as 0.0. PQ-Scene keeps the Panoptic
   Lifting rule (0.0 included) and is not touched here.

2. Loss accounting: every FN / FP is attributed to one of three causes
   using the two factors of the pair score q = IoU-bar * VC and the
   matching threshold tau, and every TP's shortfall 1 - q is split into a
   mask term (1 - IoU-bar) and a VC term IoU-bar * (1 - VC).

     FN of GT g (FP of pred p symmetric):
       A  no same-class candidate with IoU-bar > tau
       B  such a candidate exists, but none of them has VC > tau
       C  a candidate has both > tau, only the product fails (q <= tau)

   Per class with D = TP + FP/2 + FN/2 > 0 the identity
       VC-PQ + acct_vc + acct_mask + acct_thr == 1
   holds, where
       acct_vc   = (S_vc   + FN_B/2 + FP_B/2) / D
       acct_mask = (S_mask + FN_A/2 + FP_A/2) / D
       acct_thr  =          (FN_C/2 + FP_C/2) / D.

Run: .venv/bin/python -m pytest tests/test_vcpq_accounting.py
"""
import numpy as np

from vcpq.label_systems import LabelSystem
from vcpq.pq_scene_accumulator import compute_pq_from_stats
from vcpq.vcpq_accumulator import VCPQAccumulator

THING_IDS = {5, 7}
ETA = 5
LABEL = LabelSystem(name='test_things_5_7', num_classes=8, thing_ids=frozenset(THING_IDS),
                    stuff_ids=frozenset(), class_names={}, sem_lut=None)


def _run_scene(frames) -> VCPQAccumulator:
    acc = VCPQAccumulator(label=LABEL, eta=ETA)
    acc.begin_scene()
    for pred_pan, gt_pan in frames:
        acc.add_frame(pred_pan, gt_pan)
    acc.end_scene()
    return acc


def _blank():
    return np.zeros((20, 20), dtype=np.int64)


# ── 1. class averaging ───────────────────────────────────────────────────────

def _void_only_scene():
    """Class 5: one perfect TP. Class 7: no GT; one pred that has support
    (>= eta labeled px in the view) but is void-majority overall, so it is
    skipped from FP and the class ends with tp = fp = fn = 0."""
    gt = _blank()
    gt[0:10, 0:10] = 5001                       # cat-5 object; rest void
    pred = _blank()
    pred[0:10, 0:10] = 5001                     # exact match
    pred[10:20, 0:20] = 7001                    # 200 px on void
    pred[0:10, 10:13] = 7001                    # 30 px more on void → all void
    # give 7001 some labeled area so it has support but stays void-majority:
    gt[0:10, 10:13] = 5002                      # 30 labeled px (cat 5, GT #2)
    return [(pred, gt)]


def test_void_only_class_excluded_from_vcpq_mean():
    acc = _run_scene(_void_only_scene())
    s7 = acc.stats[7]
    assert s7['tp'] == 0 and s7['fp'] == 0 and s7['fn'] == 0, s7
    assert s7['fp_void_skipped'] == 1, s7
    m = acc.compute_metrics()
    # class 5: TP (5001) + FN (5002) → D = 1.5, q_sum = 1 → 2/3
    assert abs(m['per_class'][5]['vcpq'] - 2 / 3) < 1e-9, m['per_class'][5]
    # counted as 0: mean(2/3, 0.0) = 1/3. panopticapi rule: class 7 excluded → 2/3.
    assert abs(m['total']['vcpq'] - 2 / 3) < 1e-9, m['total']
    assert abs(m['things']['vcpq'] - 2 / 3) < 1e-9, m['things']
    assert m['total']['n_void_only'] == 1, m['total']


def test_pq_scene_keeps_void_only_class():
    """Regression guard: PQ-Scene averaging (Panoptic Lifting rule) unchanged."""
    stats = {
        5: dict(tp=1, fp=0, fn=1, iou_sum=1.0),
        7: dict(tp=0, fp=0, fn=0, iou_sum=0.0),
    }
    m = compute_pq_from_stats(stats, LABEL)
    assert abs(m['total']['pq'] - (2 / 3 + 0.0) / 2) < 1e-9, m['total']


# ── 2. FN / FP partition ─────────────────────────────────────────────────────

def test_identity_split_counts_as_b():
    """GT chair in 4 views, pred id 5001 in views 1-2 and 5002 in views 3-4,
    masks exact: each fragment has IoU-bar = 1, VC = 0.5 → q = 0.5 fails."""
    gt = _blank(); gt[0:10, 0:10] = 5001
    pa = _blank(); pa[0:10, 0:10] = 5001
    pb = _blank(); pb[0:10, 0:10] = 5002
    frames = [(pa, gt)] * 2 + \
             [(pb, gt)] * 2
    s = _run_scene(frames).stats[5]
    assert s['tp'] == 0 and s['fn'] == 1 and s['fp'] == 2, s
    assert (s['fn_a'], s['fn_b'], s['fn_c']) == (0, 1, 0), s
    assert (s['fp_a'], s['fp_b'], s['fp_c']) == (0, 2, 0), s


def test_low_iou_counts_as_a():
    """Pred overlaps GT with IoU 0.25 in the only view: no candidate → A."""
    gt = _blank(); gt[0:10, 0:10] = 5001                   # 100 px
    pred = _blank(); pred[5:15, 0:10] = 5001               # 50 px overlap, IoU 50/150
    s = _run_scene([(pred, gt)]).stats[5]
    assert s['tp'] == 0 and s['fn'] == 1 and s['fp'] == 1, s
    assert (s['fn_a'], s['fn_b'], s['fn_c']) == (1, 0, 0), s
    assert (s['fp_a'], s['fp_b'], s['fp_c']) == (1, 0, 0), s


def test_product_threshold_counts_as_c():
    """GT in 5 views; pred in views 1-3 with IoU 0.7, absent in 4-5:
    IoU-bar = 0.7 > tau, VC = 3/5 = 0.6 > tau, q = 0.42 <= tau → C."""
    gt = _blank(); gt[0:10, 0:10] = 5001                   # 100 px
    pred = _blank(); pred[0:7, 0:10] = 5001                # 70 px inside → IoU 0.7
    frames = [(pred, gt)] * 3 + \
             [(_blank(), gt)] * 2
    s = _run_scene(frames).stats[5]
    assert s['tp'] == 0 and s['fn'] == 1 and s['fp'] == 1, s
    assert (s['fn_a'], s['fn_b'], s['fn_c']) == (0, 0, 1), s
    assert (s['fp_a'], s['fp_b'], s['fp_c']) == (0, 0, 1), s


def test_void_skipped_pred_not_in_fp_partition():
    """fp_a + fp_b + fp_c == fp: the void-majority pred is in neither."""
    s = _run_scene(_void_only_scene()).stats[7]
    assert s['fp'] == 0 and s['fp_a'] + s['fp_b'] + s['fp_c'] == 0, s


# ── 3. TP shortfall split + identity ─────────────────────────────────────────

def test_tp_terms_and_accounting_identity():
    """One TP with IoU-bar = 1, VC = 3/4 (pred missing in view 4) plus the
    split chair (B) in the same class: per-class identity must hold."""
    gt = _blank(); gt[0:10, 0:10] = 5001; gt[10:20, 10:20] = 5003
    pa = _blank(); pa[0:10, 0:10] = 5001
    pb = _blank(); pb[0:10, 0:10] = 5002
    p3 = _blank(); p3[10:20, 10:20] = 5003
    # chair 5001: split 2/2 between pred 5001 / 5002 (B). table 5003: pred
    # 5003 present in views 1-3 only (TP, VC = 3/4).
    frames = [
        (pa + p3, gt),
        (pa + p3, gt),
        (pb + p3, gt),
        (pb,      gt),
    ]
    acc = _run_scene(frames)
    s = acc.stats[5]
    assert s['tp'] == 1 and s['fn'] == 1 and s['fp'] == 2, s
    pc = acc.compute_metrics()['per_class'][5]
    assert abs(pc['s_mask'] - 0.0) < 1e-9, pc
    assert abs(pc['s_vc'] - 0.25) < 1e-9, pc                 # 1 * (1 - 3/4)
    d = 1 + 0.5 * 2 + 0.5 * 1                                # 2.5
    assert abs(pc['acct_vc'] - (0.25 + 0.5 * 1 + 0.5 * 2) / d) < 1e-9, pc
    assert abs(pc['acct_mask'] - 0.0) < 1e-9, pc
    assert abs(pc['acct_thr'] - 0.0) < 1e-9, pc
    total = pc['vcpq'] + pc['acct_vc'] + pc['acct_mask'] + pc['acct_thr']
    assert abs(total - 1.0) < 1e-9, pc
    tot = acc.compute_metrics()['total']
    assert abs(tot['vcpq'] + tot['acct_vc'] + tot['acct_mask'] + tot['acct_thr'] - 1.0) < 1e-9, tot


def test_merge_carries_partition_counts():
    a = _run_scene([(_blank(), _blank())])                   # empty
    b = _run_scene(_void_only_scene())
    c = _run_scene([(_blank() + 0, _blank() + 0)])           # no px → nothing
    a.merge(b)
    a.merge(c)
    s = a.stats[5]
    assert s['fn'] == 1 and s['fn_a'] + s['fn_b'] + s['fn_c'] == 1, s
    assert a.stats[7]['fp_void_skipped'] == 1
