import glob
import random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import dataPreprocessing
from dataPreprocessing import draw_multirectangles
import wheatDataloader
from wheatDataloader import collate_fn
import utils
from utils import calculate_image_precision

import wandb
wandb.init(project='global_wheat')


def xyxy_xywh(boxes):
    """Change bounding boxes from xyxy to xywh.

    xywh: (x, y) is the top-left point. w: width, h: height.

    Args:
        boxes: Numpy or Tensor of shape N x 4.

    Returns:
        return: Numpy or Tensor of shape N x 4.
    """
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    boxes[:, 2] = w
    boxes[:, 3] = h
    return boxes


def format_prediction_string(boxes: np.array, scores: np.array) -> str:
    """Format boxes and scores.

    Args:
        boxes: Numpy or Tensor of shape N x 4.
        scores: Numpy or Tensor of shape N.

    Returns:
        return string in format "score1 boxes2 scores2 boxes2 ... "
    """
    # Confirm type.
    boxes = boxes.astype(np.int32)
    scores = scores.astype(np.float)
    pred_strings = []
    for score, box in zip(scores, boxes):
        format_string = f"{score:.4f} {box[0]} {box[1]} {box[2]} {box[3]}"
        pred_strings.append(format_string)
    return " ".join(pred_strings)


def eval(model_path, test_path='test'):
    """Evaluate Model. plot images

    Args:
        model_path: 'final.pth'.
        test_paht: dir contains jpg images.

    Returns:
        return result
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    eval_dataset = wheatDataloader.WheatDatasetTest(test_path)
    # Use batch_size == 1 for evaluation, DON't CHANGE
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1)
    results = []
    for images, image_ids in eval_dataloader:
        images = images.to(device)
        outputs = model(images)
        image_ids = list(map(lambda x: x.split(
            '/')[-1].strip('.jpg'), image_ids))
        for image_id, output_dict in zip(image_ids, outputs):
            boxes = output_dict['boxes'].cpu(
            ).detach().numpy()
            scores = output_dict['scores'].cpu().detach().numpy()
            # Chnage boxes from xyxy to xywh
            boxes = xyxy_xywh(boxes)
            result = {
                'image_id': image_id,
                'PredictionString': format_prediction_string(boxes, scores)
            }
            results.append(result)
    return results


if __name__ == "__main__":
    results = eval('final.pth', 'data/test/')
    print(pd.DataFrame(results))
