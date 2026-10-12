"""CLI entry points, importable from an installed console script without the
repo root needing to be on ``sys.path`` (unlike a bare ``scripts/`` module).

``vcpq-eval`` (``eval_main``) runs a ``ChunkPredictor`` over a cell, scores
its predictions right away (``vcpq.protocol.runner.evaluate_cell``) and
writes the result JSON (the ``recompute.json`` schema). With ``--dump_dir``
it also writes the cell's chunk files, which ``vcpq-recompute`` can score
again later (other ``--etas`` / ``--gammas``, no model needed).

``--predictor gt`` uses ``GroundTruthPredictor`` (a test double that returns
the GT as the "prediction" — no model needed). Any other value is
``module:factory``: an importable Python path to a function
``build(args) -> ChunkPredictor`` (``factory`` is that function's name).
If the same module defines ``add_args(parser)``, it is called first so that
the module can register its own command-line options, which then appear in
``--help`` and are parsed together with the standard ones. The current
directory is put on ``sys.path``, so a ``my_method.py`` next to where the
command runs works as ``--predictor my_method:build``.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from collections.abc import Callable

from vcpq.protocol.runner import CellSpec, GroundTruthPredictor, evaluate_cell
from vcpq.recompute import add_scoring_args, print_summary, scoring_config, write_recompute

_PREDICTOR_HELP = (
    "'gt' (returns the ground truth as the prediction; a pipeline check) or "
    "'module:factory', an importable path to a function build(args) -> ChunkPredictor "
    "(optionally with add_args(parser) in the same module), e.g. my_method:build")


def _build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a predictor over a cell and score it: VC-PQ and PQ^scene, written "
                    "to one result JSON. --dump_dir also saves the chunk files for "
                    "vcpq-recompute.")
    parser.add_argument('--predictor', required=True, help=_PREDICTOR_HELP)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--label_system', required=True, choices=['spp100', 'scannetv2_s20'])
    parser.add_argument('--frame_list', required=True)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--scenes', nargs='+', default=None)
    parser.add_argument('--out', required=True, help='result JSON (the recompute.json schema)')
    parser.add_argument('--dump_dir', default=None,
                        help='also write the chunk files (<dataset>__<scene>/c*.npz) here, '
                             'to rescore them later with vcpq-recompute')
    add_scoring_args(parser)
    parser.add_argument('--overwrite', action='store_true',
                        help='replace an existing --out file and existing chunk files under '
                             '--dump_dir')
    return parser


def _load_predictor_module(spec: str) -> tuple[Callable, Callable | None]:
    """Resolve ``module:factory`` to ``(build, add_args_or_None)``.

    Raises:
        SystemExit: malformed spec, the module cannot be imported (the message
            keeps the cause, so a missing third-party dependency inside the
            module is not reported as a missing module), or it has no such
            attribute.
    """
    module_name, sep, attr = spec.partition(':')
    if not sep or not module_name or not attr:
        raise SystemExit(
            f"--predictor {spec!r}: expected 'gt' or 'module:factory' "
            "(for example my_method:build, a function build(args) -> ChunkPredictor)")
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and (module_name == exc.name or module_name.startswith(exc.name + '.')):
            raise SystemExit(
                f"--predictor {spec!r}: module {module_name!r} not found (it must be importable "
                "from the current directory or the Python environment)") from exc
        raise SystemExit(f"--predictor {spec!r}: importing {module_name!r} failed: {exc}") from exc
    except ImportError as exc:
        raise SystemExit(f"--predictor {spec!r}: importing {module_name!r} failed: {exc}") from exc
    build = getattr(module, attr, None)
    if not callable(build):
        raise SystemExit(
            f"--predictor {spec!r}: module {module_name!r} has no callable {attr!r} "
            "(expected a function build(args) -> ChunkPredictor)")
    add_args = getattr(module, 'add_args', None)
    return build, add_args if callable(add_args) else None


def eval_main(argv: list[str] | None = None) -> None:
    """``vcpq-eval``: predict every chunk of a cell, score it, write the result JSON."""
    # The predictor module may register its own options, so it is resolved
    # before the full parse (and before --help is answered).
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument('--predictor')
    predictor_spec = pre.parse_known_args(argv)[0].predictor

    parser = _build_eval_parser()
    build = None
    if predictor_spec is not None and predictor_spec != 'gt':
        build, add_args = _load_predictor_module(predictor_spec)
        if add_args is not None:
            add_args(parser)
    args = parser.parse_args(argv)
    if args.predictor != predictor_spec:   # e.g. an abbreviated --pred, not seen by the pre-parse
        parser.error('write the option out in full: --predictor')
    if os.path.exists(args.out) and not args.overwrite:
        parser.error(f'--out {args.out} already exists (pass --overwrite to replace it)')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)   # fail now, not after the run
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    cfg = scoring_config(args)
    spec = CellSpec(
        data_root=args.data_root,
        dataset=args.dataset,
        label=cfg.label,
        frame_list=args.frame_list,
        n=args.n,
        scenes=tuple(args.scenes) if args.scenes else None,
    )
    predictor = GroundTruthPredictor(spec) if build is None else build(args)
    if not callable(getattr(predictor, 'predict', None)):
        raise SystemExit(
            f"--predictor {predictor_spec!r}: build(args) returned {type(predictor).__name__}, "
            "which has no predict(image_paths, stems, gt_size) method")
    try:
        result, _ = evaluate_cell(predictor, spec, cfg, dump_dir=args.dump_dir,
                                  overwrite=args.overwrite)
    except FileExistsError as e:
        raise SystemExit(f'--dump_dir: {e}; pass --overwrite to replace them') from None
    print_summary(result)
    write_recompute(args.out, result, None)
    print(f'Written {args.out}')
    if args.dump_dir:
        print(f'Written {result["n_chunks"]} chunk files under {args.dump_dir}')
