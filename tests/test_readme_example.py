"""The README's "Score your own predictions" examples run as written.

The code blocks marked ``<!-- readme-example:python -->`` and
``<!-- readme-example:shell -->`` in README.md are extracted, the two
placeholders ``<data_root>`` / ``<out>`` and the packaged frame-list name are
replaced by a synthetic cell (synthetic ground-truth frames written as PNGs).
The first Python block is saved as ``my_method.py`` and the ``vcpq-eval``
command is run with it through ``vcpq.cli.eval_main``, once as written and
once with ``--dump_dir`` added (the "Save chunks and rescore later"
subsection); the ``vcpq-recompute`` command then scores that directory
through ``vcpq.recompute.main`` (with at most 2 workers here) and gives the
numbers of ``vcpq-eval``. The error-decomposition example (second Python
block) reads the ``vcpq-eval`` result and prints VC-PQ and its three terms,
which sum to 1 per class and per class average. CPU only, no torch.

Run: .venv/bin/python -m pytest tests/test_readme_example.py -q
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys

import numpy as np
import pytest

from vcpq.cli import eval_main
from vcpq.protocol.dump import iter_cell, read_chunk_npz
from vcpq.recompute import main as recompute_main

from tests._helpers import build_spp100_cell
from tests.util import assert_leaves_equal

README = os.path.join(os.path.dirname(__file__), '..', 'README.md')
BLOCK = re.compile(r'<!-- readme-example:(python|shell) -->\n```\w+\n(.*?)```', re.S)
SCENE_TAG = 'scannetpp_spp100__09c1414f1b'


def _blocks(kind: str) -> list[str]:
    with open(README) as f:
        return [body for k, body in BLOCK.findall(f.read()) if k == kind]


def _fill(text: str, fixture: dict, out: str) -> str:
    text = text.replace('<data_root>', fixture['data_root']).replace('<out>', out)
    # the packaged name has 50 real scenes; point at the fixture's one-scene list
    return text.replace('--frame_list scannetpp_spp100_dslr',
                        f"--frame_list {shlex.quote(fixture['frame_list_path'])}")


def _argv(block: str, fixture: dict, out: str, command: str) -> list[str]:
    argv = shlex.split(_fill(block, fixture, out).replace('\\\n', ' '))
    assert argv[0] == command
    return argv[1:]


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _without(d: dict, *keys: str) -> dict:
    return {k: v for k, v in d.items() if k not in keys}


def test_readme_examples_run_and_score(tmp_path, capsys, monkeypatch):
    fixture = build_spp100_cell(tmp_path / 'data', n_frames=12, dataset='scannetpp_spp100')
    out = str(tmp_path / 'out')
    raw_dir = os.path.join(out, 'raw_chunks')

    py_blocks, sh_blocks = _blocks('python'), _blocks('shell')
    assert len(py_blocks) == 2 and len(sh_blocks) == 2

    # my_method.py next to the working directory, as the README runs it
    work = tmp_path / 'work'
    work.mkdir()
    (work / 'my_method.py').write_text(_fill(py_blocks[0], fixture, out))
    monkeypatch.chdir(work)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.delitem(sys.modules, 'my_method', raising=False)

    # vcpq-eval as written: one result JSON, no chunk files
    eval_argv = _argv(sh_blocks[0], fixture, out, 'vcpq-eval')
    eval_main(eval_argv)
    assert os.listdir(out) == ['results.json']
    result = _load(os.path.join(out, 'results.json'))
    assert isinstance(result['vcpq']['eta100_gamma0.0']['class_avg']['robust_filter_off']['total']['vcpq'], float)
    assert isinstance(result['pq_scene']['per_scene']['robust_filter_on'], float)
    assert (result['n_scenes'], result['n_chunks'], result['n_void_scenes']) == (1, 2, 0)
    assert result['raw_dir'] is None

    # the module's own option shows up in --help
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        eval_main(['--predictor', 'my_method:build', '--help'])
    assert e.value.code == 0 and '--checkpoint' in capsys.readouterr().out

    # "Save chunks and rescore later": the same command with --dump_dir added
    eval_main(eval_argv + ['--dump_dir', raw_dir, '--out', os.path.join(out, 'results_dump.json')])
    dumped = {t: [os.path.basename(p) for p in paths] for t, paths in iter_cell(raw_dir)}
    assert dumped == {SCENE_TAG: ['c0000.npz', 'c0001.npz']}
    c0 = read_chunk_npz(os.path.join(raw_dir, SCENE_TAG, 'c0000.npz'))
    c1 = read_chunk_npz(os.path.join(raw_dir, SCENE_TAG, 'c0001.npz'))
    assert c0.pred_pan.shape[0] == 10 and c1.pred_pan.shape[0] == 2 and c1.n == 10
    for c in (c0, c1):
        assert c.pred_pan.dtype == np.int32 and (c.pred_pan == 1000).all()
    assert sorted(c0.stems + c1.stems) == sorted(fixture['gt_by_stem'])
    with_dump = _load(os.path.join(out, 'results_dump.json'))
    assert with_dump['raw_dir'] == os.path.abspath(raw_dir)
    assert_leaves_equal(_without(result, 'raw_dir'), _without(with_dump, 'raw_dir'), rel=0)

    # vcpq-recompute on the saved chunks (workers capped at 2 to keep the test light)
    recompute_argv = _argv(sh_blocks[1], fixture, out, 'vcpq-recompute')
    i = recompute_argv.index('--workers') + 1
    recompute_argv[i] = str(min(int(recompute_argv[i]), 2))
    recompute_main(recompute_argv)
    assert os.path.isfile(os.path.join(out, 'recompute_chunks.json.gz'))
    recomputed = _load(os.path.join(out, 'recompute.json'))
    assert list(recomputed['vcpq']) == [f'eta100_gamma{g}' for g in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    # same numbers as vcpq-eval (which scored gamma 0.0 only), raw_dir included
    assert_leaves_equal(with_dump, _without(recomputed, 'chunks_file'), rel=0, subset=True)

    # the error-decomposition example on the vcpq-eval result
    terms = ('vcpq', 'acct_mask', 'acct_vc', 'acct_thr')
    block = result['vcpq']['eta100_gamma0.0']
    capsys.readouterr()
    exec(compile(_fill(py_blocks[1], fixture, out), 'README error decomposition', 'exec'), {})
    printed = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in printed] == ['total', 'things', 'stuff']
    for line, group in zip(printed, ('total', 'things', 'stuff')):
        m = block['class_avg']['robust_filter_off'][group]
        assert line == ' '.join([group] + [f'{k} {100 * m[k]:.1f}' for k in terms])
    for rf in block['class_avg']:                                # the computed state(s)
        for group in ('total', 'things', 'stuff'):
            m = block['class_avg'][rf][group]
            if m['n_classes']:
                assert abs(sum(m[k] for k in terms) - 1.0) < 1e-12, (rf, group)
        for c, pc in block['per_class'][rf].items():
            if pc['tp'] + pc['fp'] + pc['fn'] > 0:
                assert abs(sum(pc[k] for k in terms) - 1.0) < 1e-12, (rf, c)
        assert block['accounting'][rf] == {k: block['class_avg'][rf]['total'][k] for k in terms}
