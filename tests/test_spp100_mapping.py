"""Tests for scripts/spp100_mapping.py (raw ScanNet++ label -> SPP100 mapping built
at run time from the user's ScanNet++ metadata).

The synthetic test checks the mapping rule on a hand-made metadata directory. The
real-data test rebuilds the mapping from a ScanNet++ copy and checks it against
the table the paper's GT was made with (2878 rows; its SHA-256 is pinned
below). It runs only when VCPQ_SCANNETPP_ROOT
points at a ScanNet++ root (the directory holding metadata/).

Run: VCPQ_SCANNETPP_ROOT=<scannetpp_root> .venv/bin/python -m pytest tests/test_spp100_mapping.py -q
"""

from __future__ import annotations

import csv
import hashlib
import importlib.resources
import os

import pytest

from scripts.spp100_mapping import build_raw_to_spp100, label_maps, rows_to_csv

# SHA-256 of the raw label -> SPP100 table the paper's GT was made with, which
# rows_to_csv reproduces byte for byte.
EXPECTED_SHA256 = 'afed5e055b4058d41ff7f559d8356a76f25e8a7dc3115d047a5722020807fef3'


def _write_metadata(root, top100, things, raw_classes, map_rows):
    bench = root / 'semantic_benchmark'
    bench.mkdir(parents=True)
    (bench / 'top100.txt').write_text('\n'.join(top100))
    (bench / 'top100_instance.txt').write_text('\n'.join(things))
    (root / 'semantic_classes.txt').write_text('\n'.join(raw_classes) + '\n')
    with open(bench / 'map_benchmark.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class', 'semantic_map_to', 'instance_map_to'])
        w.writerows(map_rows)


def test_mapping_rule_synthetic(tmp_path):
    top100 = [f'c{i}' for i in range(100)]
    things = top100[16:]                      # c16..c99 are things (84)
    raw = ['c0', 'c20', 'wall panel', 'chairs', 'armchair', 'sofa chair', 'rug', 'odd']
    map_rows = [
        ['wall panel', 'c0', ''],             # semantic -> stuff: mapped, not crowd
        ['chairs', 'c20', 'None'],            # semantic -> thing: crowd ('None' = missing)
        ['armchair', 'c30', 'c20'],           # instance target wins
        ['sofa chair', 'armchair', ''],       # target is a raw class mapped by an earlier row
        ['rug', '', ''],                      # both missing: skipped -> void
        ['odd', 'not a class', ''],           # unknown target -> void
    ]
    _write_metadata(tmp_path, top100, things, raw, map_rows)
    rows = build_raw_to_spp100(str(tmp_path))
    assert rows == [
        ('c0', 1, 'c0', 0),
        ('c20', 21, 'c20', 0),
        ('wall panel', 1, 'c0', 0),
        ('chairs', 21, 'c20', 1),
        ('armchair', 21, 'c20', 0),
        ('sofa chair', 21, 'c20', 0),
        ('rug', 0, '', 0),
        ('odd', 0, '', 0),
    ]
    label_to_id, crowd = label_maps(rows)
    assert 'rug' not in label_to_id and label_to_id['chairs'] == 21
    assert crowd == {'chairs'}
    assert rows_to_csv(rows).splitlines()[0] == 'raw_class,spp100_id,spp100_name,iscrowd'


def test_rejects_wrong_benchmark_files(tmp_path):
    _write_metadata(tmp_path, ['a', 'b'], ['b'], ['a'], [])
    with pytest.raises(ValueError, match='unexpected semantic_benchmark'):
        build_raw_to_spp100(str(tmp_path))


@pytest.mark.skipif(not os.environ.get('VCPQ_SCANNETPP_ROOT'),
                    reason='needs VCPQ_SCANNETPP_ROOT (a ScanNet++ root with metadata/)')
def test_matches_paper_mapping():
    metadata = os.path.join(os.environ['VCPQ_SCANNETPP_ROOT'], 'metadata')
    rows = build_raw_to_spp100(metadata)
    assert len(rows) == 2878
    label_to_id, crowd = label_maps(rows)
    assert (len(label_to_id), len(crowd)) == (155, 7)
    text = rows_to_csv(rows)
    assert hashlib.sha256(text.encode()).hexdigest() == EXPECTED_SHA256

    # the ids agree with the class table the package ships
    classes = importlib.resources.files('vcpq') / 'data' / 'spp100_classes.csv'
    names = {int(r['spp100_id']): r['name']
             for r in csv.DictReader(classes.read_text().splitlines())}
    assert all(names[sid] == name for _, sid, name, _ in rows if sid > 0)
