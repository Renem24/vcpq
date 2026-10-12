"""Leaf-wise comparison of two JSON trees (recompute.json, chunk files).

Library use (tests)::

    from tests.util import assert_leaves_equal
    assert_leaves_equal(got, expected)          # ints exact, floats rel 1e-9
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Iterator


def _is_int(x: Any) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def _is_num(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def _n_leaves(x: Any) -> int:
    if isinstance(x, dict):
        return sum(_n_leaves(v) for v in x.values()) or 1
    if isinstance(x, (list, tuple)):
        return sum(_n_leaves(v) for v in x) or 1
    return 1


def diff_leaves(old: Any, new: Any, int_exact: bool = True, rel: float = 1e-9,
                subset: bool = False, path: str = '') -> Iterator[tuple[int, str]]:
    """Yield (n_leaves, message) for every difference between two JSON trees.

    Ints are compared exactly (``int_exact``), other numbers with
    ``math.isclose(rel_tol=rel)`` (``rel=0`` -> exact). A key missing on one
    side counts every leaf under it. ``subset``: keys only in ``new`` are
    ignored (``old`` is the reference).
    """
    p = path or '<root>'
    if isinstance(old, dict) and isinstance(new, dict):
        for k in old:
            if k not in new:
                yield _n_leaves(old[k]), f'{path}/{k}: missing in new'
            else:
                yield from diff_leaves(old[k], new[k], int_exact, rel, subset, f'{path}/{k}')
        if not subset:
            for k in new:
                if k not in old:
                    yield _n_leaves(new[k]), f'{path}/{k}: missing in old'
        return
    if isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
        if len(old) != len(new):
            yield max(_n_leaves(old), _n_leaves(new)), f'{p}: length {len(old)} != {len(new)}'
            return
        for i, (a, b) in enumerate(zip(old, new)):
            yield from diff_leaves(a, b, int_exact, rel, subset, f'{path}[{i}]')
        return
    if _is_num(old) and _is_num(new):
        if int_exact and _is_int(old) and _is_int(new):
            same = old == new
        elif isinstance(old, float) and isinstance(new, float) and math.isnan(old) and math.isnan(new):
            same = True
        else:
            same = math.isclose(old, new, rel_tol=rel, abs_tol=0.0)
        if not same:
            yield 1, f'{p}: {old!r} != {new!r}'
        return
    if type(old) is not type(new) and not (isinstance(old, str) and isinstance(new, str)):
        yield max(_n_leaves(old), _n_leaves(new)), f'{p}: type {type(old).__name__} != {type(new).__name__}'
        return
    if old != new:
        yield 1, f'{p}: {old!r} != {new!r}'


def assert_leaves_equal(a: Any, b: Any, int_exact: bool = True, rel: float = 1e-9,
                        subset: bool = False) -> None:
    """Assert two JSON trees are leaf-wise equal; the message names the first differing path."""
    diffs = list(diff_leaves(a, b, int_exact=int_exact, rel=rel, subset=subset))
    if diffs:
        n = sum(k for k, _ in diffs)
        raise AssertionError(f'{n} differing leaves; first: {diffs[0][1]}')

