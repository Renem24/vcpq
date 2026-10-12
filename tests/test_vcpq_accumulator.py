"""VCPQAccumulator: release API, single-view reduction.

1. compute_metrics / raw_stats flags select the six stats dicts.
2. Single view (paper Sec. 4): the view-consistency terms vanish (VC = 1 for
   every pair, so acct_vc = acct_thr = 0) and, up to the visibility threshold
   (eta = 1 here), VC-PQ equals PQ class by class.
"""
from collections import defaultdict

import numpy as np
import pytest

from tests.test_vcpq_gamma_tiling import LABEL, make_scene
from vcpq.panoptic import segments_from_pan
from vcpq.pq_scene_accumulator import compute_pq_from_stats, match_panoptic_segments
from vcpq.vcpq_accumulator import VCPQAccumulator

# (compute_metrics / raw_stats flags, accumulator attribute, recompute variant key)
VARIANTS = (
    (dict(rf=False, class_agnostic=False, things_only=False), 'stats', 'robust_filter_off'),
    (dict(rf=True, class_agnostic=False, things_only=False), 'stats_rf', 'robust_filter_on'),
    (dict(rf=False, class_agnostic=True, things_only=False), 'stats_ca', 'class_agnostic_robust_filter_off'),
    (dict(rf=True, class_agnostic=True, things_only=False), 'stats_ca_rf', 'class_agnostic_robust_filter_on'),
    (dict(rf=False, class_agnostic=True, things_only=True), 'stats_cath', 'class_agnostic_things_robust_filter_off'),
    (dict(rf=True, class_agnostic=True, things_only=True), 'stats_cath_rf', 'class_agnostic_things_robust_filter_on'),
)


def _run(label, frames, **kw) -> VCPQAccumulator:
    acc = VCPQAccumulator(label=label, **kw)
    acc.begin_scene()
    for pred_pan, gt_pan in frames:
        acc.add_frame(pred_pan, gt_pan)
    acc.end_scene()
    return acc


# ── 1. flags ─────────────────────────────────────────────────────────────────

def test_flags_select_the_six_stats_dicts():
    gt, pred = make_scene(np.random.default_rng(3))
    acc = _run(LABEL, zip(pred, gt), eta=1, class_agnostic=True)
    for flags, attr, _ in VARIANTS:
        assert acc.raw_stats(**flags) is getattr(acc, attr), attr
        assert acc.compute_metrics(**flags) == acc._compute(getattr(acc, attr)), attr
    assert acc.raw_stats(class_agnostic=True)       # collapsed pass populated
    with pytest.raises(ValueError):
        acc.compute_metrics(things_only=True)       # things-only exists only class-agnostic


# ── 2. single view ───────────────────────────────────────────────────────────

def _single_view(seed: int) -> tuple[list, list]:
    """View 0 of a random 2-view scene (make_scene's id split needs >= 2 views)."""
    gt, pred = make_scene(np.random.default_rng(seed), n_views=2)
    return gt[:1], pred[:1]


def test_single_view_vc_terms_vanish(n_seeds: int = 200):
    n_tp = 0
    for seed in range(n_seeds):
        gt, pred = _single_view(seed)
        acc = _run(LABEL, zip(pred, gt), eta=1, gamma=0.0)
        for rf in (False, True):
            for cat, pc in acc.compute_metrics(rf=rf)['per_class'].items():
                assert pc['s_vc'] == 0.0 and pc['acct_vc'] == 0.0 and pc['acct_thr'] == 0.0, (seed, cat, pc)
                assert pc['fn_b'] == pc['fn_c'] == pc['fp_b'] == pc['fp_c'] == 0, (seed, cat, pc)
                if pc['tp']:
                    assert pc['mean_c'] == 1.0, (seed, cat, pc)
                    n_tp += 1
    assert n_tp > 0


def test_single_view_reduces_to_pq(n_seeds: int = 200):
    for seed in range(n_seeds):
        gt, pred = _single_view(seed)
        vc = _run(LABEL, zip(pred, gt), eta=1, gamma=0.0).compute_metrics()['per_class']
        stats = defaultdict(lambda: dict(tp=0, fp=0, fn=0, iou_sum=0.0))
        match_panoptic_segments(pred[0], segments_from_pan(pred[0], LABEL.thing_ids),
                                gt[0], segments_from_pan(gt[0], LABEL.thing_ids), stats)
        pq = compute_pq_from_stats(stats, LABEL)['per_class']
        vc_cats = {c for c, v in vc.items() if v['tp'] + v['fp'] + v['fn'] > 0}
        pq_cats = {c for c, v in pq.items() if v['tp'] + v['fp'] + v['fn'] > 0}
        assert vc_cats == pq_cats, (seed, vc_cats, pq_cats)
        for c in vc_cats:
            assert (vc[c]['tp'], vc[c]['fp'], vc[c]['fn']) == (pq[c]['tp'], pq[c]['fp'], pq[c]['fn']), (seed, c)
            assert abs(vc[c]['vcpq'] - pq[c]['pq']) < 1e-12, (seed, c, vc[c]['vcpq'], pq[c]['pq'])
