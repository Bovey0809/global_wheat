import random
import typing
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torchvision
from matplotlib import pyplot as plt
from numba import jit
from torchvision import ops
from torchvision import transforms
from torch import optim


class EarlyStop(object):
    """Used for early stopping.

    Early stop based on steps.

    Attributes:
        steps: after N steps no changes then stop.
    """

    def __init__(self, steps):
        self.steps = steps
        self.maximum = float('-inf')
        self.losses = []

    def add(self, item):
        if len(self.losses) > self.steps:
            self.losses.pop(0)


def find_best_match(gts, pred, pred_idx, threshold=0.5, ious=None) -> int:
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):
        if gts[gt_idx][0] < 0:
            continue
        iou = -1 if ious is None else ious[gt_idx][pred_idx]
        if iou < 0:
            iou = ops.box_iou(gts[gt_idx], pred)
            if iou is not None:
                ious[gt_idx][pred_idx] = iou
        if iou < threshold:
            continue
        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx
    return best_match_idx


def calculate_precision(gts, preds, threshold=0.5, ious=None) -> float:
    n = len(preds)
    tp = 0
    fp = 0
    for pred_idx in range(n):
        best_match_gt_idx = find_best_match(
            gts, preds[pred_idx], pred_idx, threshold=threshold, ious=ious)
        if best_match_gt_idx >= 0:
            tp += 1
            gts[best_match_gt_idx] = -1
        else:
            fp += 1
    fn = (gts.sum(axis=1) > 0).sum()
    return tp / (tp + fp + fn)


def calculate_image_precision(gts, preds, thresholds=(0.5,)) -> float:
    n_threshold = len(thresholds)
    image_precision = 0.0
    # ious = np.ones((len(gts), len(preds))) * -1
    ious = ops.box_iou(gts, preds).detach().cpu().numpy()
    gts = gts.cpu().numpy()
    for threshold in thresholds:
        precition_at_threshold = calculate_precision(
            gts.copy(), preds, threshold=threshold, ious=ious)
        image_precision += precition_at_threshold / n_threshold
    return image_precision


def get_transforms(tsfms: List[str]):
    transform_list = []
    for tsfm in tsfms:
        if tsfm == 'Normalize':
            transform_list.append(
                transforms.Normalize((0, 0, 0), (1, 1, 1)))
        elif tsfm == 'RandomRotation':
            transform_list.append(transforms.RandomRotation(90))
        elif tsfm == 'RandomResizedCrop':
            transform_list.append(transforms.RandomResizedCrop(512))
        else:
            transform_list.append(getattr(transforms, tsfm)())
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def get_optimizer(optimizer):
    optimizer = getattr(optim, optimizer)
    return optimizer


if __name__ == "__main__":
    validation_image_precision = []
    iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]
