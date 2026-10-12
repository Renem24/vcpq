"""Raw ScanNet++ label -> SPP100 class mapping, built from the user's ScanNet++ copy.

preprocess_scannetpp.py calls build_raw_to_spp100() at run time instead of
reading a shipped table. The inputs are the ScanNet++ release files

    <scannetpp_root>/metadata/semantic_classes.txt                 (raw labels, row order)
    <scannetpp_root>/metadata/semantic_benchmark/top100.txt        (100 classes, id = line + 1)
    <scannetpp_root>/metadata/semantic_benchmark/top100_instance.txt  (84 thing classes)
    <scannetpp_root>/metadata/semantic_benchmark/map_benchmark.csv (raw -> benchmark name)

Mapping rule (the rule the paper's GT was made with; it follows the mapping loop
of PanSt3R's tools/preprocess_scannetpp.py, https://github.com/naver/panst3r):
    1. top100 names get ids 1..100 (void = 0).
    2. Rows of map_benchmark.csv are visited in file order. A row whose
       instance_map_to and semantic_map_to are both missing is skipped. Otherwise
       the raw class takes the id of instance_map_to when that name already has
       an id, else the id of semantic_map_to when that name has one. A raw class
       mapped through semantic_map_to onto a thing class is flagged iscrowd.
       "Already has an id" includes raw classes mapped by earlier rows.
    3. Every raw class of semantic_classes.txt that got no id maps to 0 (void).

map_benchmark.csv is read with the csv module; the cells that pandas.read_csv
treats as missing by default (for example the string "None" in
instance_map_to) are treated as missing here too.

No third-party dependencies, so the tests import it without the rendering stack.
"""

from __future__ import annotations

import csv
import io
import os.path as osp

# pandas.read_csv default na_values (pandas/_libs/parsers.pyx STR_NA_VALUES).
PANDAS_NA_STRINGS = frozenset({
    '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
    '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a',
    'nan', 'null',
})

CSV_HEADER = ('raw_class', 'spp100_id', 'spp100_name', 'iscrowd')


def _read_lines(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _cell(value: str | None) -> str | None:
    return None if value is None or value in PANDAS_NA_STRINGS else value


def build_raw_to_spp100(metadata_dir: str) -> list[tuple[str, int, str, int]]:
    """Rows (raw_class, spp100_id, spp100_name, iscrowd), one per line of
    semantic_classes.txt in file order; spp100_id 0 = void, spp100_name '' then.

    Args:
        metadata_dir: <scannetpp_root>/metadata
    """
    bench = osp.join(metadata_dir, 'semantic_benchmark')
    names = _read_lines(osp.join(bench, 'top100.txt'))
    thing_names = set(_read_lines(osp.join(bench, 'top100_instance.txt')))
    raw_classes = _read_lines(osp.join(metadata_dir, 'semantic_classes.txt'))
    if len(names) != 100 or len(thing_names) != 84 or not thing_names <= set(names):
        raise ValueError(f'unexpected semantic_benchmark files in {bench}: '
                         f'{len(names)} top100 names, {len(thing_names)} instance names')

    label_to_id = {name: i + 1 for i, name in enumerate(names)}
    crowd = set()
    with open(osp.join(bench, 'map_benchmark.csv'), newline='') as f:
        for row in csv.DictReader(f):
            cls_name = _cell(row['class'])
            sem_target = _cell(row['semantic_map_to'])
            inst_target = _cell(row['instance_map_to'])
            if sem_target is None and inst_target is None:
                continue
            if sem_target not in label_to_id and inst_target not in label_to_id:
                continue
            if inst_target is not None and inst_target in label_to_id:
                label_to_id[cls_name] = label_to_id[inst_target]
            elif sem_target is not None and sem_target in label_to_id:
                label_to_id[cls_name] = label_to_id[sem_target]
                if sem_target in thing_names:
                    crowd.add(cls_name)

    id_to_name = {i + 1: n for i, n in enumerate(names)}
    rows = []
    for raw in raw_classes:
        spp100_id = label_to_id.get(raw, 0)
        rows.append((raw, spp100_id, id_to_name.get(spp100_id, ''), int(raw in crowd)))
    return rows


def rows_to_csv(rows: list[tuple[str, int, str, int]]) -> str:
    """The rows as CSV text (csv.writer defaults, header first)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_HEADER)
    writer.writerows(rows)
    return buf.getvalue()


def label_maps(rows: list[tuple[str, int, str, int]]) -> tuple[dict[str, int], set[str]]:
    """(label_to_id for the mapped raw classes only, raw classes flagged iscrowd)."""
    label_to_id = {raw: sid for raw, sid, _, _ in rows if sid > 0}
    crowd = {raw for raw, _, _, iscrowd in rows if iscrowd == 1}
    return label_to_id, crowd
