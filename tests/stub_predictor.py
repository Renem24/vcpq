"""Stub predictor modules for the ``vcpq-eval --predictor module:factory``
tests, CPU only.

``build``: "predicts" a constant class for every pixel (option ``--sem_id``).
``build_noisy``: an imperfect, view-inconsistent prediction of the ground
truth (``NoisyPredictor``); every instance it builds is kept in ``built`` so
that a test can check its ``predict`` calls.
"""

from __future__ import annotations

import os

import numpy as np

from vcpq.label_systems import LabelSystem, load_label_system
from vcpq.protocol.gt import load_gt
from vcpq.protocol.runner import ChunkPrediction


class ConstPredictor:
    def __init__(self, sem_id: int):
        self.sem_id = sem_id

    def predict(self, image_paths, stems, gt_size):
        return ChunkPrediction([np.full(gt_size, self.sem_id * 1000, np.int32) for _ in stems])


def add_args(parser):
    parser.add_argument('--sem_id', type=int, default=1)


def build(args):
    return ConstPredictor(args.sem_id)


class NoisyPredictor:
    """The ground truth shifted right by 2 to 5 pixels (depending on the
    stem), with the thing instance ids of odd-numbered stems renumbered
    (``inst % 3 + 1``), so that classes score strictly between 0 and 1 and
    some objects change id between views. ``predicted`` records every
    ``predict`` call as (scene, stems)."""

    def __init__(self, label: LabelSystem):
        self.label = label
        self.predicted: list[tuple[str, tuple[str, ...]]] = []

    def predict(self, image_paths, stems, gt_size):
        scene_dir = os.path.dirname(os.path.dirname(image_paths[0]))
        self.predicted.append((os.path.basename(scene_dir), tuple(stems)))
        pred_pans = []
        for stem in stems:
            pan = np.roll(load_gt(scene_dir, stem, self.label), 2 + int(stem) % 4, axis=1)
            if int(stem) % 2:
                inst = pan % 1000
                pan = np.where(inst > 0, pan - inst + inst % 3 + 1, pan)
            pred_pans.append(pan.astype(np.int32))
        return ChunkPrediction(pred_pans)


built: list[NoisyPredictor] = []


def build_noisy(args):
    predictor = NoisyPredictor(load_label_system(args.label_system))
    built.append(predictor)
    return predictor


def not_a_predictor(args):
    return object()


NOT_CALLABLE = 3
