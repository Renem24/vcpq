"""vcpq.vocab: the open-vocabulary prompt lists of the two label systems
(torch-free). Pins lengths, ends and the mapping to class ids."""

from __future__ import annotations

import csv
import dataclasses
import importlib.resources

import pytest

from vcpq.label_systems import S20_20_CLASS_NAMES, load_label_system
from vcpq.vocab import S20_OTHERFURNITURE_ID, build_vocab


def test_s20_vocab_is_41_prompts():
    prompts, idx = build_vocab(load_label_system('scannetv2_s20'))
    csv_text = (importlib.resources.files('vcpq') / 'data' / 's20_200_to_20.csv').read_text()
    constituents = [row[1] for row in csv.reader(csv_text.splitlines())
                    if row and int(row[-1]) == S20_OTHERFURNITURE_ID]
    assert len(prompts) == 41 and len(idx) == 41
    assert prompts[:19] == [S20_20_CLASS_NAMES[k] for k in range(1, 20)]
    assert prompts[19:] == constituents and len(constituents) == 22
    assert idx == list(range(1, 20)) + [20] * 22
    assert prompts[0] == S20_20_CLASS_NAMES[1] and prompts[-1] == constituents[-1]
    assert 'otherfurniture' not in prompts
    # literal strings: first, last concrete class, first/second and last id-20 prompt
    assert (prompts[0], prompts[18], prompts[19], prompts[20], prompts[40]) == (
        'wall', 'bathtub', 'tv stand', 'trash can', 'keyboard piano')


def test_spp100_vocab_is_100_names_in_id_order():
    label = load_label_system('spp100')
    prompts, idx = build_vocab(label)
    assert len(prompts) == 100 and len(idx) == 100
    assert prompts == [label.class_names[k] for k in range(1, 101)]
    assert idx == list(range(1, 101))
    assert prompts[0] == label.class_names[1] and prompts[-1] == label.class_names[100]
    assert (prompts[0], prompts[19], prompts[99]) == ('wall', 'doorframe', 'marker')


def test_unknown_label_system_raises():
    other = dataclasses.replace(load_label_system('spp100'), name='other')
    with pytest.raises(ValueError, match='other'):
        build_vocab(other)
