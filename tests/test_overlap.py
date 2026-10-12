"""Overlap table == per-pair boolean masks.

The matching counterpart (test_pq_matching_matches_reference,
match_panoptic_segments) is in tests/test_pq_scene.py.

Run: .venv/bin/python -m pytest tests/test_overlap.py -v
"""
import numpy as np

from vcpq.overlap import panoptic_overlap_table


# ── reference implementation: per-pair boolean masks ─────────────────────────
def naive_table(pred_pan, gt_pan, pred_ids, gt_ids):
    void_mask = gt_pan == 0
    pred_area, pred_void = [], []
    for pid in pred_ids:
        pmask = pred_pan == pid
        pred_area.append(int(pmask.sum()))
        pred_void.append(int((pmask & void_mask).sum()))
    gt_area = [int((gt_pan == gid).sum()) for gid in gt_ids]
    inter = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    for gi, gid in enumerate(gt_ids):
        gmask = gt_pan == gid
        for pi, pid in enumerate(pred_ids):
            inter[gi, pi] = int((gmask & (pred_pan == pid)).sum())
    return inter, np.array(gt_area), np.array(pred_area), np.array(pred_void)


# ── case generator ───────────────────────────────────────────────────────────
def random_case(rng, h=40, w=52, n_cls=4, n_inst=3, void_frac=0.25,
                pred_shift=3, drop_pred=0.2, extra_pred=2):
    """A (pred, gt) map pair with voids, duplicates and unmatched segments."""
    gt = np.zeros((h, w), dtype=np.int32)
    ids = []
    for c in range(1, n_cls + 1):
        for i in range(1, n_inst + 1):
            y, x = rng.integers(0, h - 8), rng.integers(0, w - 8)
            gt[y:y + rng.integers(3, 9), x:x + rng.integers(3, 9)] = c * 1000 + i
            ids.append(c * 1000 + i)
    gt[rng.random((h, w)) < void_frac] = 0          # scatter void
    gt_segments = [dict(id=int(v), category_id=int(v) // 1000,
                        isthing=int(v) // 1000 <= 2)
                   for v in np.unique(gt) if v != 0]

    pred = np.zeros((h, w), dtype=np.int32)
    for s in gt_segments:
        if rng.random() < drop_pred:
            continue
        m = np.roll(np.roll(gt == s['id'], rng.integers(0, pred_shift + 1), 0),
                    rng.integers(0, pred_shift + 1), 1)
        pred[m] = s['id']
    for j in range(extra_pred):                     # predictions with no GT
        y, x = rng.integers(0, h - 6), rng.integers(0, w - 6)
        pred[y:y + 5, x:x + 5] = rng.integers(1, n_cls + 1) * 1000 + 900 + j
    pred_segments = [dict(id=int(v), category_id=int(v) // 1000,
                          isthing=int(v) // 1000 <= 2)
                     for v in np.unique(pred) if v != 0]
    return pred, pred_segments, gt, gt_segments


def test_overlap_table_equals_masks():
    rng = np.random.default_rng(0)
    n = 0
    for _ in range(40):
        pred, pseg, gt, gseg = random_case(rng)
        pids = [s['id'] for s in pseg]
        gids = [s['id'] for s in gseg]
        for pl, gl in ((pids, gids),
                       (pids, [g for g in gids if g // 1000 <= 2]),   # things only
                       ([], gids), (pids, []),                        # empty lists
                       (pids + [999999], gids)):                      # id absent from map
            t = panoptic_overlap_table(pred, gt, pl, gl)
            ri, rg, rp, rv = naive_table(pred, gt, pl, gl)
            assert np.array_equal(t.inter, ri), 'intersections differ'
            assert np.array_equal(t.gt_area, rg), 'gt areas differ'
            assert np.array_equal(t.pred_area, rp), 'pred areas differ'
            assert np.array_equal(t.pred_void, rv), 'void overlaps differ'
            n += 1
    # degenerate maps
    z = np.zeros((7, 9), dtype=np.int32)
    t = panoptic_overlap_table(z, z, [], [])
    assert t.inter.shape == (0, 0) and t.gt_area.size == 0
    print(f'[ok] overlap table == per-pair boolean masks ({n} tables + empty map)')


if __name__ == '__main__':
    test_overlap_table_equals_masks()
    print('ALL OK')
