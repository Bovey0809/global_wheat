import random
import typing
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from numba import jit
from torch import optim as optim
from torchvision import ops
import albumentations as transforms
from albumentations.pytorch.transforms import ToTensorV2

import wandb


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


# def calculate_precision(gts, preds, threshold=0.5, ious=None) -> float:
#     n = len(preds)
#     tp = 0
#     fp = 0
#     for pred_idx in range(n):
#         best_match_gt_idx = find_best_match(
#             gts, preds[pred_idx], pred_idx, threshold=threshold, ious=ious)
#         if best_match_gt_idx >= 0:
#             tp += 1
#             gts[best_match_gt_idx] = -1
#         else:
#             fp += 1
#     fn = (gts.sum(axis=1) > 0).sum()
#     return tp / (tp + fp + fn)


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
        if tsfm == 'RandomCrop':
            transform_list.append(
                transforms.RandomCrop(512, 512))
        elif tsfm == 'RandomSizedCrop':
            transform_list.append(
                transforms.RandomSizedCrop((500, 800), 512, 512)
            )
        else:
            transform_list.append(getattr(transforms, tsfm)())
    transform_list.append(ToTensorV2())
    return transforms.Compose(transform_list, bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_optimizer(optimizer):
    optimizer = getattr(optim, optimizer)
    return optimizer


def calculate_precision(boxes_true: torch.tensor, boxes_pred: torch.tensor, confidences: list, threshold=0.5) -> float:
    confidences = confidences.cpu().numpy()
    if boxes_true.size(1) == 0:
        return 0
    iou = ops.box_iou(boxes_pred, boxes_true)
    pr_matches = set()
    gt_matches = set()

    match_candidates = (iou >= threshold).nonzero()
    GT_PR_matches = defaultdict(list)
    for PR, GT in match_candidates:
        GT_PR_matches[GT.item()].append(PR.item())
    for GT, PRs in GT_PR_matches.items():
        if len(PRs) > 1:
            pr_match = PRs[confidences[PRs].argsort()[-1]]
        else:
            pr_match = PRs[0]
        if pr_match not in pr_matches:
            gt_matches.add(GT)
        pr_matches.add(pr_match)
    TP = len(pr_matches)
    pr_idx = range(iou.size(0))
    gt_idx = range(iou.size(1))

    FP = len(set(pr_idx).difference(pr_matches))
    FN = len(set(gt_idx).difference(gt_matches))

    return TP / (TP + FP + FN)


def calculate_mean_precision(boxes_true, boxes_pred, confidences, thresholds=(0.5,)):
    precision = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        precision[i] = calculate_precision(
            boxes_true, boxes_pred, confidences, threshold=threshold)
    return precision.mean()


def create_box_data(boxes, labels=None, scores=None, caption_prefix='image', iteration=0):
    """Create a dict for boxes inorder to draw or used for json.

    Args:
        boxes: xyxy format boxes of shape N x 4.
        caption_prefix: the prefix for the image.

    Returns:
        return dictionary
    """
    if labels is None:
        labels = [1] * len(boxes)
    if scores is None:
        scores = [1] * len(boxes)
    box_data = []
    for idx, box in enumerate(boxes):
        box_data.append({
            "position": {"minX": int(box[0]), "maxX": int(box[2]), "minY": int(box[1]), "maxY": int(box[3])},
            "class_id": labels[idx],
            "domain": "pixel",
            "box_caption": f"{caption_prefix}: iter {iteration} box: {idx}",
            "scores": {"confidence": float(scores[idx])}})
    return box_data


def draw_image(image, box_data, caption='image_id', group='predictions'):
    """Draw boxes on images and assigned to a group.

    Args:
        image: PIL image.
        box_data: dict create from func "create_box_data" or create by yourself.
        caption: The unique name for the image.
        group: the group to use to show images.

    Returns:
        retun wandb.Image object.
    """
    boxes_dict = {f"{group}": {"box_data": box_data}}
    return wandb.Image(image, caption=caption, boxes=boxes_dict)


def assert_same(config, hyperparameters):
    for key1, key2 in zip(config.keys(), hyperparameters.keys()):
        assert config[key1] == hyperparameters[key2]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    boxes_true = torch.tensor([
        [0., 0., 10., 10.],
        [0., 0., 12., 10.]
    ])
    boxes_pred = torch.tensor([
        [0., 0., 10., 6.],
        [0, 0, 10, 5]
    ])
    confidence = [.5, .9]
    score = calculate_precision(boxes_true, boxes_pred, confidence)
    print(score)
    confidence = [.9, .5]
    score = calculate_precision(boxes_true, boxes_pred, confidence)
    print(score)
