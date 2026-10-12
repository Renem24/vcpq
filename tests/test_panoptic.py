import numpy as np, pytest
from vcpq.panoptic import encode_panoptic, segments_from_pan

def test_encode_things_and_stuff():
    sem = np.array([[0, 5, 5], [1, 1, 5]], np.uint8)
    inst = np.array([[0, 2, 0], [0, 0, 3]], np.uint16)
    pan = encode_panoptic(sem, inst, frozenset({5}))
    assert pan.dtype == np.int32
    assert pan.tolist() == [[0, 5002, 0], [1000, 1000, 5003]]

def test_encode_rejects_inst_overflow():
    with pytest.raises(ValueError):
        encode_panoptic(np.full((1, 1), 5, np.uint8), np.full((1, 1), 1000, np.int64), frozenset({5}))

def test_segments_roundtrip():
    pan = np.array([[5002, 1000], [0, 5003]], np.int32)
    segs = segments_from_pan(pan, frozenset({5}))
    assert segs == [{'id': 1000, 'category_id': 1, 'isthing': False},
                    {'id': 5002, 'category_id': 5, 'isthing': True},
                    {'id': 5003, 'category_id': 5, 'isthing': True}]

def test_encode_rejects_negative_inst():
    with pytest.raises(ValueError, match='negative inst id'):
        encode_panoptic(np.full((1, 2), 5, np.uint8), np.array([[1, -1]], np.int64), frozenset({5}))
    # the check is global: a negative id on a stuff pixel is rejected too
    with pytest.raises(ValueError, match='negative inst id'):
        encode_panoptic(np.array([[1]], np.uint8), np.array([[-2]], np.int64), frozenset({5}))


# ── validate_pred_pan ───────────────────────────────────────────────────────

def _label():
    from vcpq.label_systems import load_label_system
    return load_label_system('spp100')     # 84 thing, 16 stuff classes, 1..100


def _thing_stuff():
    label = _label()
    return label, min(label.thing_ids), min(label.stuff_ids)


def test_validate_accepts_valid_map():
    from vcpq.panoptic import validate_pred_pan
    label, th, st = _thing_stuff()
    validate_pred_pan(np.array([[0, st * 1000], [th * 1000 + 1, th * 1000 + 999]], np.int32), label)
    validate_pred_pan(np.zeros((2, 2), np.int32), label)


@pytest.mark.parametrize('case,match', [
    ('negative', 'negative panoptic ids: \\[-3\\]'),
    ('unknown_class', 'class not in label system .*\\[150001\\].*\\[150\\]'),
    ('void_with_inst', 'class 0 other than 0.*\\[7\\]'),
    ('stuff_with_inst', 'stuff panoptic ids with a nonzero instance id.*STUFF'),
    ('thing_without_inst', 'thing panoptic ids without an instance id.*THING'),
])
def test_validate_rejects(case, match):
    from vcpq.panoptic import validate_pred_pan
    label, th, st = _thing_stuff()
    bad = {'negative': -3, 'unknown_class': 150001, 'void_with_inst': 7,
           'stuff_with_inst': st * 1000 + 4, 'thing_without_inst': th * 1000}[case]
    pan = np.array([[0, st * 1000, th * 1000 + 1, bad]], np.int32)
    match = match.replace('STUFF', str(st * 1000 + 4)).replace('THING', str(th * 1000))
    with pytest.raises(ValueError, match=match):
        validate_pred_pan(pan, label)
