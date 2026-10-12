"""JSON blocks of ``recompute.json`` (split out of ``vcpq.recompute``).

One block per metric family from a pooled (class-avg) accumulator and the
per-scene value lists: ``vcpq_block`` (per (eta, gamma)) and ``pq_block``
(PQ^scene).
"""

from __future__ import annotations

import numpy as np

from vcpq.pq_scene_accumulator import PQSceneAccumulator
from vcpq.vcpq_accumulator import VCPQAccumulator


def _mean(xs: list) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _grp(m: dict) -> dict:
    """total/things/stuff of a compute_metrics() result, scalar fields only."""
    return {g: {k: v for k, v in m[g].items() if isinstance(v, (int, float))}
            for g in ('total', 'things', 'stuff') if g in m}


def _ca_tot(m: dict) -> dict:
    """Scalar fields of a class-agnostic ``total`` (the single class-agnostic number)."""
    return {k: v for k, v in m['total'].items() if isinstance(v, (int, float))}


def _pc(m: dict) -> dict:
    """``per_class`` of a compute_metrics() result with string class keys (JSON)."""
    return {str(c): dict(v) for c, v in sorted(m['per_class'].items())}


def _ca_block(a, ps: dict[str, list], key: str, class_agnostic: bool) -> dict:
    """class_avg / per_scene / accounting / class_agnostic / per_class of a VC-PQ
    or PQ^scene accumulator, for the robust-filtering states in ``ps``
    (``robust_filter_off`` / ``robust_filter_on`` -> per-scene values)."""
    rfs = {name: name == 'robust_filter_on' for name in ps}
    metrics = {name: a.compute_metrics(rf=rf) for name, rf in rfs.items()}
    block = {
        'class_avg': {name: _grp(m) for name, m in metrics.items()},
        'per_scene': {name: _mean(values) for name, values in ps.items()},
    }
    if key == 'vcpq':   # 1 - VC-PQ split (macro), from class-avg total
        keys = ('vcpq', 'acct_vc', 'acct_mask', 'acct_thr')
        block['accounting'] = {name: {k: m['total'].get(k) for k in keys}
                               for name, m in metrics.items()}
    if class_agnostic:   # class head removed from matching
        ca = {name: a.compute_metrics(rf=rf, class_agnostic=True) for name, rf in rfs.items()}
        cath = {name: a.compute_metrics(rf=rf, class_agnostic=True, things_only=True)
                for name, rf in rfs.items()}   # things-only (stuff voided)
        block['class_agnostic'] = {
            **{name: _ca_tot(m) for name, m in ca.items()},
            **{f'things_{name}': _ca_tot(m) for name, m in cath.items()},
            'per_class': {**{name: _pc(m) for name, m in ca.items()},
                          **{f'things_{name}': _pc(m) for name, m in cath.items()}},
        }
    # pooled per-class raw counts + derived values (exact pooling / accounting)
    block['per_class'] = {name: _pc(m) for name, m in metrics.items()}
    return block


def vcpq_block(a: VCPQAccumulator, ps: dict[str, list], class_agnostic: bool) -> dict:
    """JSON block of one VC-PQ (eta, gamma) accumulator + its per-scene lists."""
    return _ca_block(a, ps, 'vcpq', class_agnostic)


def pq_block(a: PQSceneAccumulator, ps: dict[str, list], class_agnostic: bool) -> dict:
    """JSON block of the PQ^scene accumulator + its per-scene lists."""
    return _ca_block(a, ps, 'pq', class_agnostic)
