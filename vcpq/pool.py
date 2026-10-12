"""Pool the result JSONs of several cells into one (``vcpq-pool``).

The paper's ScanNet++ tables pool the DSLR and the iPhone cell of a method
(99 scene-runs). ``pool_results`` gives the numbers that scoring the chunks
of all input cells together would give:

  - class averages (VC-PQ with its error decomposition, PQ^scene class-avg):
    the per-class raw counts of the inputs are added and the metrics are
    computed again from the sums, as for one cell;
  - per-scene means (the paper's PQ^scene, per-scene VC-PQ): the mean over
    the scenes of all inputs, i.e. the inputs' means weighted by the number
    of scenes that entered them.

The inputs must share the label system, the ``include_void_scenes`` option,
the robust-filter threshold, the (eta, gamma) blocks and the computed
robust-filtering states (``--both_robust_filter``). The pooled JSON has the keys of a result JSON
except ``raw_dir``, the error-decomposition repeat ``accounting`` and the
class-agnostic blocks.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable

from vcpq.label_systems import load_label_system
from vcpq.pq_scene_accumulator import PQSceneAccumulator
from vcpq.recompute import print_summary
from vcpq.recompute_blocks import _grp, _pc
from vcpq.vcpq_accumulator import VCPQAccumulator

# result key -> (compute_metrics rf flag, accumulator stats attribute)
_RF = {'robust_filter_off': (False, 'stats'), 'robust_filter_on': (True, 'stats_rf')}


def _n_in_means(res: dict) -> int:
    """Number of scenes that entered the per-scene means of one result."""
    return res['n_scenes'] - (0 if res['include_void_scenes'] else res['n_void_scenes'])


def _pooled_mean(results: list[dict], value: Callable[[dict], float]) -> float:
    """Mean over the scenes of all results, from each result's per-scene mean."""
    n = [_n_in_means(r) for r in results]
    return sum(value(r) * k for r, k in zip(results, n)) / sum(n) if sum(n) else 0.0


def _pool_family(acc, results: list[dict], block: Callable[[dict], dict]) -> dict:
    """``class_avg`` / ``per_scene`` / ``per_class`` of one metric family.

    Args:
        acc: An empty ``VCPQAccumulator`` or ``PQSceneAccumulator``; the
            per-class raw counts of every result are added into its robust_filter_off /
            robust_filter_on stats.
        results: The result JSONs.
        block: Picks the family's block out of one result.
    """
    states = list(block(results[0])['per_class'])          # the computed robust_filter_off / robust_filter_on
    for res in results:
        for rf in states:
            stats = getattr(acc, _RF[rf][1])
            for cat, s in block(res)['per_class'][rf].items():
                dst = stats[int(cat)]
                for k in dst:
                    dst[k] += s[k]
    metrics = {rf: acc.compute_metrics(rf=_RF[rf][0]) for rf in states}
    return {
        'class_avg': {rf: _grp(m) for rf, m in metrics.items()},
        'per_scene': {rf: _pooled_mean(results, lambda r: block(r)['per_scene'][rf])
                      for rf in states},
        'per_class': {rf: _pc(m) for rf, m in metrics.items()},
    }


def pool_results(results: list[dict]) -> dict:
    """Pool result JSONs (``vcpq-eval`` / ``vcpq-recompute``) of several cells.

    Raises:
        ValueError: The results differ in label system, ``include_void_scenes``,
            ``robust_filter_threshold``, (eta, gamma) blocks or computed robust-filtering
            states, or one was written without
            ``include_void_scenes`` / ``robust_filter_threshold`` (an older version of
            this code).
    """
    for i, r in enumerate(results):
        for key in ('include_void_scenes', 'robust_filter_threshold'):
            if key not in r:
                raise ValueError(f'result {i} has no {key} key; score it again with '
                                 f'this version of vcpq')
    for key in ('label_system', 'include_void_scenes', 'robust_filter_threshold'):
        if len({r[key] for r in results}) != 1:
            raise ValueError(f'the results differ in {key}: {[r[key] for r in results]}')
    blocks = [list(r['vcpq']) for r in results]
    if any(b != blocks[0] for b in blocks):
        raise ValueError(f'the results differ in their (eta, gamma) blocks: {blocks}')
    states = [(list(r['pq_scene']['per_class']), list(r['vcpq'][blocks[0][0]]['per_class']))
              for r in results]
    if any(s != states[0] for s in states):
        raise ValueError(f'the results differ in their robust-filtering states (--both_robust_filter): {states}')

    label = load_label_system(results[0]['label_system'])
    return {
        'label_system': label.name,
        'n_scenes': sum(r['n_scenes'] for r in results),
        'n_chunks': sum(r['n_chunks'] for r in results),
        'n_void_scenes': sum(r['n_void_scenes'] for r in results),
        'include_void_scenes': results[0]['include_void_scenes'],
        'robust_filter_threshold': results[0]['robust_filter_threshold'],
        'pq_scene': _pool_family(PQSceneAccumulator(label), results, lambda r: r['pq_scene']),
        'vcpq': {key: _pool_family(VCPQAccumulator(label), results, lambda r: r['vcpq'][key])
                 for key in blocks[0]},
    }


def main(argv: list[str] | None = None) -> None:
    """``vcpq-pool``: pool result JSONs, e.g. the ScanNet++ DSLR and iPhone cells."""
    ap = argparse.ArgumentParser(
        prog='vcpq-pool',
        description="Pool the result JSONs of several cells into one, as the paper's ScanNet++ "
                    "tables pool the DSLR and the iPhone cell of a method.")
    ap.add_argument('results', nargs='+',
                    help='result JSONs of vcpq-eval or vcpq-recompute (same label system and options)')
    ap.add_argument('--out', required=True, help='pooled JSON')
    args = ap.parse_args(argv)

    results = []
    for path in args.results:
        with open(path) as f:
            results.append(json.load(f))
    pooled = {'inputs': [os.path.abspath(p) for p in args.results], **pool_results(results)}
    print_summary(pooled)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(pooled, f, indent=2)
    print(f'Written {args.out}')
