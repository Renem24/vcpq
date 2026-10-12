from vcpq.label_systems import load_label_system

def test_spp100_counts():
    ls = load_label_system('spp100')
    assert ls.num_classes == 101 and len(ls.thing_ids) == 84 and len(ls.stuff_ids) == 16
    assert ls.thing_ids.isdisjoint(ls.stuff_ids) and ls.sem_lut is None

def test_s20_lut():
    ls = load_label_system('scannetv2_s20')
    assert ls.thing_ids == frozenset(range(3, 21)) and ls.stuff_ids == frozenset({1, 2})
    assert ls.sem_lut.shape == (201,) and ls.sem_lut[0] == 0 and ls.sem_lut.max() == 20

def test_unknown_raises():
    import pytest
    with pytest.raises(ValueError): load_label_system('unknown')

def test_label_system_hashable_identity():
    a, b = load_label_system('scannetv2_s20'), load_label_system('scannetv2_s20')
    assert len({a, b}) == 2 and a == a and a != b     # identity, no field-wise hash
