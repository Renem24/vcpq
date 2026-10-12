"""PQSceneAccumulator (tiling PQ^scene), match_panoptic_segments, compute_pq_from_stats.

1. Tiling: the views of a chunk are stacked into one [S*H, W] image and
   matched once, so a segment id that recurs across views is ONE segment
   whose IoU is taken over the tiled masks (cases a-b); the things-only
   class-agnostic pass voids stuff before matching (c); the robust filter
   drops a rare class from the robust_filter_on variant only (d).
2. match_panoptic_segments == a reference matching that uses per-pair
   boolean masks instead of the overlap table.

Run: .venv/bin/python -m pytest tests/test_pq_scene.py
"""
from collections import defaultdict

import numpy as np
import pytest

from tests.test_overlap import random_case
from tests.test_vcpq_accumulator import VARIANTS
from tests.test_vcpq_gamma_tiling import LABEL, make_scene
from vcpq.pq_scene_accumulator import (PQSceneAccumulator, _void_stuff, compute_pq_from_stats,
                                       match_panoptic_segments)

H, W = 20, 20


def _blank() -> np.ndarray:
    return np.zeros((H, W), dtype=np.int64)


def _run(frames, **kw) -> PQSceneAccumulator:
    """One chunk: frames = [(pred_pan, gt_pan), ...]."""
    acc = PQSceneAccumulator(label=LABEL, **kw)
    acc.begin_scene()
    for pred_pan, gt_pan in frames:
        acc.add_frame(pred_pan, gt_pan)
    acc.end_scene()
    return acc


# ── 1. tiling ────────────────────────────────────────────────────────────────

def test_same_id_across_views_is_one_tiled_segment():
    """(a) GT 5001 = 100 px in each of 2 views; pred 5001 exact in view 0 and
    on half of the view-1 mask -> one TP whose IoU is (100 + 50) / 200."""
    g0 = _blank(); g0[0:10, 0:10] = 5001
    g1 = _blank(); g1[5:15, 5:15] = 5001
    p0 = g0.copy()
    p1 = _blank(); p1[5:10, 5:15] = 5001
    acc = _run([(p0, g0), (p1, g1)])
    assert dict(acc.raw_stats()) == {5: dict(tp=1, fp=0, fn=0, iou_sum=150 / 200)}
    assert acc.compute_metrics()['per_class'][5]['pq'] == 0.75


@pytest.mark.parametrize('rows0,expect', [
    # larger fragment 120/200 > 0.5 -> TP; the 80 px fragment (80/200) -> FP
    (12, dict(tp=1, fp=1, fn=0, iou_sum=120 / 200)),
    # both fragments 100/200 = 0.5, not > 0.5 -> FN + 2 FP
    (10, dict(tp=0, fp=2, fn=1, iou_sum=0.0)),
])
def test_id_switch_across_views(rows0, expect):
    """(b) Same GT object (200 px over 2 views); the prediction is exact in
    both views but switches id 5001 -> 5002 in view 1."""
    g0 = _blank(); g0[0:rows0, 0:10] = 5001
    g1 = _blank(); g1[0:20 - rows0, 0:10] = 5001
    p0 = g0.copy()
    p1 = np.where(g1 == 5001, 5002, 0)
    assert dict(_run([(p0, g0), (p1, g1)]).raw_stats()) == {5: expect}


def test_things_only_voids_stuff_first():
    """(c) GT thing 5001 next to a stuff wall 9000; the predicted thing covers
    both. Class-agnostic with stuff: IoU 100/200 against either GT segment,
    no match. Things-only: the wall is voided in GT (and pred), so the 100
    predicted pixels on it leave the union -> IoU 100/100."""
    gt = _blank(); gt[0:10, 0:10] = 5001; gt[0:10, 10:20] = 9000
    pred = _blank(); pred[0:10, 0:20] = 5001
    acc = _run([(pred, gt)], class_agnostic=True)
    assert dict(acc.raw_stats(class_agnostic=True)) == {-1: dict(tp=0, fp=1, fn=2, iou_sum=0.0)}
    assert dict(acc.raw_stats(class_agnostic=True, things_only=True)) == {-1: dict(tp=1, fp=0, fn=0, iou_sum=1.0)}
    assert np.array_equal(_void_stuff(gt, LABEL.thing_ids), np.where(gt == 9000, 0, gt))


def _rare_class_chunk():
    """2 views x 400 px non-void; class 7 has 3 GT pixels (3/800 < 0.005) and
    is missed by the prediction."""
    g0 = np.full((H, W), 5001, dtype=np.int64)
    g1 = np.full((H, W), 5002, dtype=np.int64); g1[0, 0:3] = 7001
    p1 = g1.copy(); p1[0, 0:3] = 0
    return [(g0.copy(), g0), (p1, g1)]


def test_robust_filter_drops_rare_class():
    """(d) rf off scores class 7 (FN, PQ 0 in the mean); rf on voids it."""
    acc = _run(_rare_class_chunk())
    off, on = acc.compute_metrics(), acc.compute_metrics(rf=True)
    assert sorted(off['per_class']) == [5, 7] and off['total']['pq'] == 0.5
    assert sorted(on['per_class']) == [5] and on['total']['pq'] == 1.0


def test_rf_threshold_zero_disables_filter():
    """robust_filter_threshold=0 means robust filtering off (robust_filter_on == robust_filter_off)."""
    acc = _run(_rare_class_chunk(), robust_filter_threshold=0)
    assert dict(acc.raw_stats(rf=True)) == dict(acc.raw_stats())


def test_flags_select_the_six_stats_dicts():
    gt, pred = make_scene(np.random.default_rng(3))
    acc = _run(zip(pred, gt), class_agnostic=True)
    for flags, attr, _ in VARIANTS:
        assert acc.raw_stats(**flags) is getattr(acc, attr), attr
        assert acc.compute_metrics(**flags) == compute_pq_from_stats(getattr(acc, attr), LABEL), attr
    assert acc.raw_stats(class_agnostic=True)       # collapsed pass populated
    with pytest.raises(ValueError):
        acc.compute_metrics(things_only=True)       # things-only exists only class-agnostic


def test_frame_buffers_are_public_and_reset():
    gt, pred = make_scene(np.random.default_rng(4), n_views=3)
    acc = PQSceneAccumulator(label=LABEL)
    acc.begin_scene()
    for g, p in zip(gt, pred):
        acc.add_frame(p, g)
    assert len(acc.pred_pans) == len(acc.gt_pans) == 3 and acc.pred_pans[1] is pred[1]
    acc.end_scene()
    assert acc.pred_pans == [] and acc.gt_pans == []
    before = {c: dict(s) for c, s in acc.raw_stats().items()}
    acc.end_scene()                                 # empty chunk: no-op
    assert {c: dict(s) for c, s in acc.raw_stats().items()} == before


# ── 2. match_panoptic_segments == reference matching with boolean masks ─────

def match_reference(pred_pan, pred_segments, gt_pan, gt_segments, stats,
                    void_threshold=0.5):
    """match_panoptic_segments computed with per-pair boolean masks (no overlap table)."""
    gt_matched, pred_matched = set(), set()
    void_mask = gt_pan == 0
    pred_info = {}
    for pseg in pred_segments:
        pid = pseg['id']
        pmask = pred_pan == pid
        parea = int(pmask.sum())
        if parea == 0:
            continue
        pred_info[pid] = (pmask, parea, int((pmask & void_mask).sum()))
    matches = []
    for pseg in pred_segments:
        pid, pcat = pseg['id'], pseg['category_id']
        if pid not in pred_info:
            continue
        pmask, parea, pred_void = pred_info[pid]
        for gseg in gt_segments:
            gid, gcat = gseg['id'], gseg['category_id']
            if pcat != gcat:
                continue
            gmask = gt_pan == gid
            inter = int((pmask & gmask).sum())
            if inter == 0:
                continue
            union = (parea - pred_void) + int(gmask.sum()) - inter
            if union <= 0:
                continue
            iou = inter / union
            if iou > 0.5:
                matches.append((iou, pid, gid, pcat))
    matches.sort(key=lambda x: -x[0])
    for iou, pid, gid, cat in matches:
        if pid in pred_matched or gid in gt_matched:
            continue
        pred_matched.add(pid)
        gt_matched.add(gid)
        stats[cat]['tp'] += 1
        stats[cat]['iou_sum'] += iou
    for pseg in pred_segments:
        pid = pseg['id']
        if pid in pred_matched or pid not in pred_info:
            continue
        _, parea, pred_void = pred_info[pid]
        void_frac = pred_void / parea if parea > 0 else 0.0
        if void_frac > void_threshold:
            continue
        stats[pseg['category_id']]['fp'] += 1
    for gseg in gt_segments:
        if gseg['id'] not in gt_matched:
            stats[gseg['category_id']]['fn'] += 1


def test_pq_matching_matches_reference():
    rng = np.random.default_rng(7)
    for i in range(60):
        pred, pseg, gt, gseg = random_case(
            rng, pred_shift=int(rng.integers(0, 5)),
            drop_pred=float(rng.random() * 0.4),
            void_frac=float(rng.random() * 0.6))
        a = defaultdict(lambda: dict(tp=0, fp=0, fn=0, iou_sum=0.0))
        b = defaultdict(lambda: dict(tp=0, fp=0, fn=0, iou_sum=0.0))
        match_reference(pred, pseg, gt, gseg, a)
        match_panoptic_segments(pred, pseg, gt, gseg, b)
        assert sorted(a) == sorted(b), f'class sets differ (case {i})'
        for c in a:
            assert a[c]['tp'] == b[c]['tp'] and a[c]['fp'] == b[c]['fp'] \
                and a[c]['fn'] == b[c]['fn'], f'counts differ (case {i}, class {c})'
            assert a[c]['iou_sum'] == b[c]['iou_sum'], f'iou_sum differs (case {i})'
