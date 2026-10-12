"""Label systems (thing/stuff/class-name definitions) for VC-PQ evaluation.

Two closed sets ship with this package: ScanNet++ top100 fused panoptic
(spp100, no on-disk remap needed) and ScanNetv2 s20 (20-class, remapped at
eval time from the 200-class on-disk labels via ``sem_lut``).
"""

from __future__ import annotations

import csv
import importlib.resources
from dataclasses import dataclass

import numpy as np

SPP100 = 'spp100'
SCANNETV2_S20 = 'scannetv2_s20'

# The 20 ScanNet benchmark class names by class id (0 is void).
S20_20_CLASS_NAMES: dict[int, str] = {
    1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 13: "desk", 14: "curtain",
    15: "refrigerator", 16: "shower curtain", 17: "toilet", 18: "sink",
    19: "bathtub", 20: "otherfurniture",
}


@dataclass(frozen=True, eq=False)
class LabelSystem:
    """A closed panoptic label space: names, thing/stuff split, disk remap.

    Compared and hashed by identity (``eq=False``): the fields hold a dict and
    an array, which a field-wise ``__hash__`` could not hash.

    Attributes:
        name: 'spp100' or 'scannetv2_s20'.
        num_classes: Number of continuous ids including void (index 0).
        thing_ids: Semantic ids that carry instances.
        stuff_ids: Semantic ids with no instance concept.
        class_names: {continuous id: class name}.
        sem_lut: [201] int32 on-disk (200-class) -> continuous-id remap for
            s20; None for spp100 (already stored in its own continuous ids).
    """

    name: str
    num_classes: int
    thing_ids: frozenset[int]
    stuff_ids: frozenset[int]
    class_names: dict[int, str]
    sem_lut: np.ndarray | None


def _read_csv_lines(filename: str) -> list[str]:
    path = importlib.resources.files('vcpq') / 'data' / filename
    return path.read_text().splitlines()


def _load_spp100() -> LabelSystem:
    """spp100_classes.csv: header `spp100_id,name,isthing`."""
    class_names: dict[int, str] = {}
    thing_ids: set[int] = set()
    stuff_ids: set[int] = set()
    for row in csv.DictReader(_read_csv_lines('spp100_classes.csv')):
        cid = int(row['spp100_id'])
        class_names[cid] = row['name']
        (thing_ids if int(row['isthing']) else stuff_ids).add(cid)
    return LabelSystem(
        name=SPP100,
        num_classes=101,
        thing_ids=frozenset(thing_ids),
        stuff_ids=frozenset(stuff_ids),
        class_names=class_names,
        sem_lut=None,
    )


def _load_s20() -> LabelSystem:
    """s20_200_to_20.csv: rows `cont_200,name,cont_20`, no header."""
    lut = np.zeros(201, dtype=np.int32)
    for line in _read_csv_lines('s20_200_to_20.csv'):
        if not line.strip():
            continue
        cont_200, _name, cont_20 = line.strip().split(',')
        lut[int(cont_200)] = int(cont_20)
    return LabelSystem(
        name=SCANNETV2_S20,
        num_classes=21,
        thing_ids=frozenset(range(3, 21)),
        stuff_ids=frozenset({1, 2}),
        class_names=dict(S20_20_CLASS_NAMES),
        sem_lut=lut,
    )


_LOADERS = {SPP100: _load_spp100, SCANNETV2_S20: _load_s20}


def load_label_system(name: str) -> LabelSystem:
    """Load a LabelSystem by name.

    Raises:
        ValueError: If `name` is not one of the registered label systems.
    """
    loader = _LOADERS.get(name)
    if loader is None:
        raise ValueError(f'unknown label system {name!r}; expected one of {sorted(_LOADERS)}')
    return loader()
