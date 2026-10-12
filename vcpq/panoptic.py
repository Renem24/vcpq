"""Panoptic id encoding: pan = sem * SEM_MULT + inst, 0 = void.

Things carry an instance id (``sem_id * 1000 + inst_id``); an instance id of 0
on a thing class means "no instance" and encodes to void (0), not to the bare
stuff-style ``sem_id * 1000``. Stuff classes have no instance concept and
always encode to ``sem_id * 1000``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vcpq.label_systems import LabelSystem

SEM_MULT: int = 1000


def encode_panoptic(sem: np.ndarray, inst: np.ndarray, thing_ids: frozenset[int]) -> np.ndarray:
    """Encode (sem, inst) maps into one panoptic id map.

    Args:
        sem: [H, W] semantic class id (uint8/int), 0 = void.
        inst: [H, W] instance id (int), 0 = no instance.
        thing_ids: Semantic ids that carry instances.

    Returns:
        [H, W] int32 panoptic map: things -> sem * SEM_MULT + inst (inst == 0
        -> 0), stuff -> sem * SEM_MULT, void (sem == 0) -> 0.

    Raises:
        ValueError: If any ``inst >= SEM_MULT`` (would corrupt the encoding)
            or any ``inst < 0``. Both checks cover every pixel of ``inst``,
            including stuff and void pixels whose instance id is otherwise
            ignored.
    """
    sem = np.asarray(sem)
    inst = np.asarray(inst)
    if inst.size and int(inst.max()) >= SEM_MULT:
        raise ValueError(f'inst id >= {SEM_MULT} cannot be encoded (got max {int(inst.max())})')
    if inst.size and int(inst.min()) < 0:
        raise ValueError(f'negative inst id cannot be encoded (got min {int(inst.min())})')

    is_thing = np.isin(sem, np.fromiter(thing_ids, dtype=sem.dtype, count=len(thing_ids)))
    pan = sem.astype(np.int64) * SEM_MULT
    pan = np.where(is_thing, np.where(inst > 0, pan + inst, 0), pan)
    pan = np.where(sem == 0, 0, pan)
    return pan.astype(np.int32)


def validate_pred_pan(pan: np.ndarray, label: LabelSystem) -> None:
    """Check that a predicted panoptic map is a valid encoding in ``label``.

    Rules: every id is ``>= 0``; every class ``pan // SEM_MULT`` is 0 (void)
    or one of ``label.thing_ids`` / ``label.stuff_ids``; a void pixel is
    exactly 0 (ids 1 to ``SEM_MULT - 1`` are rejected); a stuff pixel has
    instance id 0 (``pan % SEM_MULT == 0``); a thing pixel has an instance
    id from 1 to ``SEM_MULT - 1``. The scorer does not check its input, so a
    map that breaks a rule is scored silently wrong (an unknown class enters
    the class averages, an id 1 to 999 becomes a segment of class 0, a stuff
    class with instance ids splits into several segments, a negative id
    fails deep inside the overlap count). ``vcpq-eval``
    (``vcpq.protocol.runner.iter_scene_chunks``) and ``write_chunk_npz`` with
    ``label`` call this; reading a dump never does.

    Args:
        pan: [H, W] (or any shape) integer panoptic map, ``sem * SEM_MULT + inst``.
        label: The label system the map is scored in.

    Raises:
        ValueError: A rule is broken; the message names the offending ids.
    """
    ids = np.unique(np.asarray(pan)).astype(np.int64)
    neg = ids[ids < 0]
    if neg.size:
        raise ValueError(f'negative panoptic ids: {neg[:10].tolist()}')
    sem = ids // SEM_MULT
    inst = ids % SEM_MULT
    known = np.fromiter(sorted({0} | label.thing_ids | label.stuff_ids), dtype=np.int64)
    bad = ids[~np.isin(sem, known)]
    if bad.size:
        raise ValueError(f'panoptic ids with a class not in label system {label.name!r}: '
                         f'{bad[:10].tolist()} (classes {sorted(set((bad // SEM_MULT).tolist()))[:10]})')
    bad = ids[(sem == 0) & (ids != 0)]
    if bad.size:
        raise ValueError(f'panoptic ids of class 0 other than 0 (void must be exactly 0): '
                         f'{bad[:10].tolist()}')
    stuff = np.fromiter(sorted(label.stuff_ids), dtype=np.int64)
    bad = ids[np.isin(sem, stuff) & (inst != 0)]
    if bad.size:
        raise ValueError(f'stuff panoptic ids with a nonzero instance id (stuff must be '
                         f'class_id * {SEM_MULT}): {bad[:10].tolist()}')
    things = np.fromiter(sorted(label.thing_ids), dtype=np.int64)
    bad = ids[np.isin(sem, things) & (inst == 0)]
    if bad.size:
        raise ValueError(f'thing panoptic ids without an instance id (a thing pixel needs '
                         f'an instance id from 1 to {SEM_MULT - 1}): {bad[:10].tolist()}')


def segments_from_pan(pan: np.ndarray, thing_ids: frozenset[int]) -> list[dict]:
    """Re-derive segment dicts from a panoptic map (id = sem * SEM_MULT + inst)."""
    out = []
    for pid in np.unique(pan):
        pid = int(pid)
        if pid == 0:
            continue
        sem = pid // SEM_MULT
        out.append(dict(id=pid, category_id=sem, isthing=sem in thing_ids))
    return out
