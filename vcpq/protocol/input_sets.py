"""The input-set protocol: partitions a scene's frame stems into the fixed-size
multi-view input sets a chunk is evaluated on, plus the packaged view lists
(scene -> 120 frame stems) that fix which frames form every input set in the
paper's tables.

``make_input_sets`` builds the paper's input sets: shuffle the scene's frame
indices with ``random.Random(seed)``, slice into fixed-size chunks in
shuffled order, then sort each chunk back into stem order (numeric when the
stems are digit strings) before scoring it. ``view_offset`` is the slice
start ``i`` into the *shuffled* index array, not a position in stem order,
so a dumped chunk's ``view_offset`` can be looked up back against
``make_input_sets``' output.

The three packaged lists correspond to the paper's scene pools:
``scannetpp_spp100_dslr`` (50 scenes), ``scannetpp_spp100_iphone`` (50 scenes,
one of which — ``d755b3d9d8`` — has an entirely void ground truth and is
left out of the reported per-scene means, see the README), and
``scannetv2_s20`` (312 scenes).
"""

from __future__ import annotations

import importlib.resources
import json
import logging
import random

logger = logging.getLogger('vcpq.protocol.input_sets')

_PACKAGED: dict[str, str] = {
    'scannetpp_spp100_dslr': 'scannetpp_spp100_dslr_120f_seed42.json',
    'scannetpp_spp100_iphone': 'scannetpp_spp100_iphone_120f_seed42.json',
    'scannetv2_s20': 'scannetv2_s20_120f_seed42.json',
}


def load_frame_list(name_or_path: str) -> dict[str, list[str]]:
    """Load a scene -> frame-stems list.

    ``name_or_path`` is one of the three packaged names in ``_PACKAGED``
    (loaded as a package resource) or a filesystem path to an equivalent JSON
    ``{scene: [stems]}``.
    """
    filename = _PACKAGED.get(name_or_path)
    if filename is not None:
        resource = importlib.resources.files('vcpq.protocol') / 'lists' / filename
        return json.loads(resource.read_text())
    with open(name_or_path) as f:
        return json.load(f)


def _sort_key(stem: str):
    return int(stem) if stem.isdigit() else stem


def make_input_sets(stems: list[str], n: int, seed: int = 42) -> list[tuple[int, list[str]]]:
    """Partition ``stems`` into input sets of (up to) ``n`` frames.

    Shuffle ``range(len(stems))`` with ``random.Random(seed)``, slice the
    shuffled indices into consecutive ``n``-sized chunks, sort each chunk back
    to stem order. If ``len(stems) <= n`` the scene is scored as a single set
    (offset 0) of all frames. If ``len(stems) > n`` and ``n`` does not divide
    it, the last set holds the remaining ``len(stems) % n`` frames (a warning
    is logged). The packaged 120-frame lists never hit this for
    N = 5, 10, 20, 40.

    Returns:
        List of ``(view_offset, stems)`` in chunk order, where
        ``view_offset`` is the slice start in the *shuffled* index array.
    """
    if len(stems) <= n:
        return [(0, list(stems))]
    if len(stems) % n:
        logger.warning('%d frames, N=%d: the last input set has %d frames (fewer than N)',
                       len(stems), n, len(stems) % n)
    indices = list(range(len(stems)))
    random.Random(seed).shuffle(indices)
    sets: list[tuple[int, list[str]]] = []
    for i in range(0, len(stems), n):
        chunk = indices[i:i + n]
        chunk.sort(key=lambda j: _sort_key(stems[j]))
        sets.append((i, [stems[j] for j in chunk]))
    return sets
