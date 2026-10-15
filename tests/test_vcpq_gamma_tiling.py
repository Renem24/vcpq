"""VC-PQ gamma=1 (area-weighted pair score) vs PQ-Scene tiling.

Claim under test (vcpq_accumulator.py gamma):

    With gamma = 1 the pair score becomes
        q_1(p, g) = sum_v inter_v / sum_v union_v   over v in V(p) ∪ V(g),
    i.e. the IoU of the two masks tiled over the images of the support union.
    This is exactly the tiling IoU used by PQ-Scene, restricted to the images
    where the instance passes the visibility threshold eta.

Tests:
  1. eta = 1, gamma = 1  →  per-class {tp, fp, fn, q_sum} of VCPQAccumulator
     equals {tp, fp, fn, iou_sum} of PQSceneAccumulator (robust_filter_off AND robust_filter_on)
     on random synthetic scenes. With eta = 1 every non-empty appearance is
     visible, so the support restriction is void and the identity is exact.
  2. eta > 1, gamma = 1  →  on scenes reduced to one candidate pair per
     class, the matched pair's q (the class's q_sum) equals the brute-force
     concat IoU over the support union; its i (mean IoU, i_sum) and c (VC,
     c_sum) stay equal-image, so gamma changes q only.
  3. A toy counterexample (areas 1000/20/20/20, only the large image
     predicted): gamma = 0 gives q = 0.25 (no match), gamma = 1 gives
     q = 1000/1060 (match).

Run: .venv/bin/python -m pytest tests/test_vcpq_gamma_tiling.py
"""
import numpy as np

from vcpq.label_systems import LabelSystem
from vcpq.pq_scene_accumulator import PQSceneAccumulator
from vcpq.vcpq_accumulator import VCPQAccumulator

THING_IDS = {5, 7}
STUFF_IDS = {9}
H, W = 24, 32
LABEL = LabelSystem(name='test_things_5_7_stuff_9', num_classes=10, thing_ids=frozenset(THING_IDS),
                    stuff_ids=frozenset(STUFF_IDS), class_names={}, sem_lut=None)


def _rect(rng: np.random.Generator, hmin: int, hmax: int) -> tuple[slice, slice]:
    h = int(rng.integers(hmin, hmax + 1))
    w = int(rng.integers(hmin, hmax + 1))
    y = int(rng.integers(0, H - h + 1))
    x = int(rng.integers(0, W - w + 1))
    return slice(y, y + h), slice(x, x + w)


def make_scene(rng: np.random.Generator, n_views: int = 4):
    """Random multi-view GT/pred panoptic maps with splits, drops, shifts,
    spurious predictions, class errors, tiny appearances and GT void patches.

    Returns:
        gt_pans, pred_pans: lists of [H, W] int64 maps (0 = void / no pred).
    """
    gt = [np.zeros((H, W), dtype=np.int64) for _ in range(n_views)]
    n_inst = int(rng.integers(3, 7))
    inst = []
    for k in range(n_inst):
        cat = int(rng.choice([5, 7, 9]))
        iid = cat * 1000 + (1 if cat == 9 else k + 1)   # stuff: a single instance
        inst.append(iid)
        for v in range(n_views):
            if rng.random() < 0.7:
                ys, xs = _rect(rng, 1, 10)      # side 1..10 → areas 1..100 (tiny ones included)
                gt[v][ys, xs] = iid
    # GT void patches (annotation holes)
    for v in range(n_views):
        if rng.random() < 0.5:
            ys, xs = _rect(rng, 2, 6)
            gt[v][ys, xs] = 0

    pred = [g.copy() for g in gt]
    for iid in inst:
        cat = iid // 1000
        # identity split: from a random view on, use a new id
        if rng.random() < 0.3:
            v0 = int(rng.integers(1, n_views))
            new_id = cat * 1000 + 500 + (iid % 1000)
            for v in range(v0, n_views):
                pred[v][pred[v] == iid] = new_id
        # drop in one view
        if rng.random() < 0.2:
            v = int(rng.integers(0, n_views))
            pred[v][pred[v] == iid] = 0
        # shift in one view
        if rng.random() < 0.4:
            v = int(rng.integers(0, n_views))
            m = pred[v] == iid
            if m.any():
                pred[v][m] = 0
                m2 = np.roll(m, shift=(int(rng.integers(-3, 4)), int(rng.integers(-3, 4))), axis=(0, 1))
                pred[v][m2] = iid
        # class error
        if rng.random() < 0.15:
            new_cat = 7 if cat == 5 else 5
            for v in range(n_views):
                pred[v][pred[v] == iid] = new_cat * 1000 + (iid % 1000)
    # spurious predictions (possibly on void)
    for v in range(n_views):
        if rng.random() < 0.5:
            ys, xs = _rect(rng, 1, 8)
            cat = int(rng.choice([5, 7]))
            pred[v][ys, xs] = cat * 1000 + 900 + v
    return gt, pred


def _run_vcpq(gt, pred, eta: int, gamma: float) -> VCPQAccumulator:
    acc = VCPQAccumulator(label=LABEL, eta=eta, gamma=gamma)
    acc.begin_scene()
    for g, p in zip(gt, pred):
        acc.add_frame(p, g)
    acc.end_scene()
    return acc


def _run_pqs(gt, pred):
    acc = PQSceneAccumulator(label=LABEL)
    acc.begin_scene()
    for g, p in zip(gt, pred):
        acc.add_frame(p, g)
    acc.end_scene()
    return acc


def _stats_equal(vc: dict, pq: dict, tol: float = 1e-9) -> tuple[bool, str]:
    cats = set(vc.keys()) | set(pq.keys())
    for c in sorted(cats):
        a = vc.get(c, dict(tp=0, fp=0, fn=0, q_sum=0.0))
        b = pq.get(c, dict(tp=0, fp=0, fn=0, iou_sum=0.0))
        if (a['tp'], a['fp'], a['fn']) != (b['tp'], b['fp'], b['fn']):
            return False, f'cat {c}: vcpq tp/fp/fn={a["tp"]}/{a["fp"]}/{a["fn"]} pqs={b["tp"]}/{b["fp"]}/{b["fn"]}'
        if abs(a['q_sum'] - b['iou_sum']) > tol:
            return False, f'cat {c}: q_sum {a["q_sum"]:.12f} vs iou_sum {b["iou_sum"]:.12f}'
    return True, ''


# ── 1. eta=1, gamma=1 ≡ PQ-Scene (robust_filter_off and robust_filter_on) ─────────────────────────

def test_gamma1_eta1_equals_pq_scene(n_seeds: int = 400):
    n_nontrivial = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        gt, pred = make_scene(rng)
        vc = _run_vcpq(gt, pred, eta=1, gamma=1.0)
        pq = _run_pqs(gt, pred)
        ok, msg = _stats_equal(vc.stats, pq.raw_stats())
        assert ok, f'seed {seed} robust_filter_off: {msg}'
        ok, msg = _stats_equal(vc.stats_rf, pq.raw_stats(rf=True))
        assert ok, f'seed {seed} robust_filter_on: {msg}'
        if any(s['tp'] > 0 for s in vc.stats.values()):
            n_nontrivial += 1
    assert n_nontrivial > n_seeds // 2, n_nontrivial
    print(f'[1] eta=1 gamma=1 == PQ-Scene on {n_seeds} scenes ({n_nontrivial} with >=1 TP), robust_filter_off+robust_filter_on')


# ── 2. gamma=1 pair score = concat IoU over the support union (eta>1) ────────

def _brute_force_q1(gt, pred, gid: int, pid: int, eta: int):
    """Concat IoU of (pid, gid) over V(p) ∪ V(g); void-free pred areas."""
    inter_sum = union_sum = 0
    for g, p in zip(gt, pred):
        gm = g == gid
        pm = (p == pid) & (g != 0)          # void convention: pred px on GT void removed
        g_vis = gm.sum() >= eta
        p_vis = pm.sum() >= eta
        if not (g_vis or p_vis):
            continue
        ga = int(gm.sum()) if g_vis else 0
        pa = int(pm.sum()) if p_vis else 0
        inter = int((gm & pm).sum()) if (g_vis and p_vis) else 0
        inter_sum += inter
        union_sum += ga + pa - inter
    return inter_sum / union_sum if union_sum > 0 else 0.0


def _brute_force_tiling_iou(gt, pred, gid: int, pid: int):
    gs = np.vstack(gt)
    ps = np.vstack(pred)
    gm = gs == gid
    pm = (ps == pid) & (gs != 0)
    inter = int((gm & pm).sum())
    union = int(gm.sum()) + int(pm.sum()) - inter
    return inter / union if union > 0 else 0.0


def _one_pair_per_class(gt, pred):
    """Keep, per class, only the smallest GT id, in the GT and in the prediction;
    every other id becomes 0. Each class then has one candidate pair (the id
    with itself), so a matched class's q_sum / i_sum / c_sum are that pair's
    q / i / c."""
    kept: dict[int, int] = {}
    for gid in sorted({int(i) for g in gt for i in np.unique(g) if i}):
        kept.setdefault(gid // 1000, gid)
    ids = list(kept.values())
    return ([np.where(np.isin(g, ids), g, 0) for g in gt],
            [np.where(np.isin(p, ids), p, 0) for p in pred], ids)


def test_gamma1_pair_score_is_concat_iou_over_support_union(n_seeds: int = 300, eta: int = 6):
    n_pairs = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        gt, pred, ids = _one_pair_per_class(*make_scene(rng))
        acc1 = _run_vcpq(gt, pred, eta=eta, gamma=1.0)
        acc0 = _run_vcpq(gt, pred, eta=eta, gamma=0.0)
        acc_e1 = _run_vcpq(gt, pred, eta=1, gamma=1.0)
        for gid in ids:
            cat = gid // 1000
            s1, s0, se1 = acc1.stats.get(cat), acc0.stats.get(cat), acc_e1.stats.get(cat)
            if s1 is not None and s1['tp'] == 1:
                q_bf = _brute_force_q1(gt, pred, gid, gid, eta)
                assert abs(s1['q_sum'] - q_bf) < 1e-12, (seed, gid, s1, q_bf)
                n_pairs += 1
                # i and c are equal-image in both runs: same pair → same i, c
                if s0 is not None and s0['tp'] == 1:
                    assert abs(s1['i_sum'] - s0['i_sum']) < 1e-12 and \
                        abs(s1['c_sum'] - s0['c_sum']) < 1e-12, (seed, gid, s1, s0)
                    assert abs(s0['q_sum'] - s0['i_sum'] * s0['c_sum']) < 1e-12, s0
            # eta=1: q is the plain tiling IoU over all images
            if se1 is not None and se1['tp'] == 1:
                assert abs(se1['q_sum'] - _brute_force_tiling_iou(gt, pred, gid, gid)) < 1e-12, \
                    (seed, gid, se1)
    assert n_pairs > 0
    print(f'[2] gamma=1 q == concat IoU over support union on {n_pairs} matched pairs (eta={eta}); i, c unchanged')


# ── 3. Toy counterexample ───────────────────────────────────────────────────

def _counterexample_scene():
    Hc, Wc = 40, 40
    gt, pred = [], []
    areas = [(25, 40), (4, 5), (4, 5), (4, 5)]   # 1000, 20, 20, 20 px
    for v, (h, w) in enumerate(areas):
        g = np.zeros((Hc, Wc), dtype=np.int64)
        g[:h, :w] = 5001
        gt.append(g)
        pred.append(g.copy() if v == 0 else np.zeros((Hc, Wc), dtype=np.int64))
    return gt, pred


def test_counterexample_1000_20_20_20():
    gt, pred = _counterexample_scene()
    for gamma, q_expect, tp, fn, fp in ((0.0, 0.25, 0, 1, 1), (1.0, 1000 / 1060, 1, 0, 0)):
        acc = VCPQAccumulator(label=LABEL, eta=5, gamma=gamma)
        acc.begin_scene()
        for g, p in zip(gt, pred):
            acc.add_frame(p, g)
        acc.end_scene()
        s = acc.stats[5]
        assert (s['tp'], s['fn'], s['fp']) == (tp, fn, fp), (gamma, s)
        if tp:                                   # one pair: q_sum is its q
            assert abs(s['q_sum'] - q_expect) < 1e-12


def test_counterexample_1000_20_20_20_pq_scene():
    gt, pred = _counterexample_scene()
    pq = _run_pqs(gt, pred).raw_stats()[5]
    assert (pq['tp'], pq['fn'], pq['fp']) == (1, 0, 0) and abs(pq['iou_sum'] - 1000 / 1060) < 1e-12, pq
    print('[3] counterexample 1000/20/20/20: gamma=0 q=0.25 (FN+FP), gamma=1 q=1000/1060 (TP) == PQ-Scene')

