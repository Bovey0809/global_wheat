import wandb
from wheatDataloader import collate_fn
import wheatDataloader
import dataPreprocessing
from matplotlib import pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision import ops
import utils
from utils import calculate_image_precision
import PIL
from PIL import Image
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox

hyperparameter_defaults = dict(
    batch_size=16,
    learning_rate=0.005,
    epochs=4,
    frac=0.999,  # train / (train + val)
    seed=0,
    num_workers=0,
    # To tensor is added to the last as default.
    transforms=['Flip'],
    # 'RandomGrayscale', 'RandomRotation', 'RandomVerticalFlip'],
    optimizer='SGD',  # SGD is also support,
    lr_step=2000
)

wandb.init(project="global_wheat", config=hyperparameter_defaults,
           tags=['sgd'])
print(hyperparameter_defaults)
DATA_DIR = 'data'
torch.backends.cudnn.benchmark = True
utils.assert_same(wandb.config, hyperparameter_defaults)


def train(model, dataloader, optimizer, epoch, device, lr_scheduler):
    model.train(True)
    for iteration, (images, targets, images_ids) in enumerate(dataloader, 0):
        images = images.to(device)
        targets = [{'boxes': i['boxes'].to(
            device), 'labels':i['labels'].to(device)} for i in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler:
            lr_scheduler.step()

        for k, v in loss_dict.items():
            wandb.log({'_'.join(['Train', k]): v})
        wandb.log({"Train_loss": loss})
        if iteration % 50 == 0:
            print(
                f"epoch:{epoch} iter: {iteration} loss:{loss.item()} lr: {utils.get_lr(optimizer)}")


def test(model, dataloder, epoch, device):
    model.eval()
    test_loss = 0
    best_loss = 100
    id2labels = {0: "background", 1: "wheat"}
    with torch.no_grad():
        for iteration, (images, targets, images_ids) in enumerate(dataloder, 0):
            images = images.to(device)
            targets = [{'boxes': i['boxes'].to(
                device), 'labels':i['labels'].to(device)} for i in targets]
            # precision
            predictions = model(images)
            batch_precision = 0
            for i, prediction in enumerate(predictions):
                boxes_pred = prediction['boxes']
                scores = prediction['scores']
                boxes_true = targets[i]['boxes']
                thresholds = np.arange(0.5, 0.75, 0.05)
                mean_precision = utils.calculate_mean_precision(
                    boxes_true, boxes_pred, scores, thresholds)
                # wandb.log({"validation_mean_precision": mean_precision})
                batch_precision += mean_precision

                # Visulize image with precision
                boxes_true = boxes_true.cpu().numpy()
                boxes_pred = boxes_pred.cpu().numpy()
                scores = scores.cpu().numpy()

                box_data_pred = utils.create_box_data(
                    boxes_pred, scores=scores, caption_prefix='Validation_pred', iteration=epoch)
                box_data_true = utils.create_box_data(
                    boxes_true, caption_prefix='Validation_GT', iteration=epoch)
                boxes_dict = {"predictions": {"box_data": box_data_pred, "class_labels": id2labels},
                              "ground_truth": {"box_data": box_data_true, "class_labels": id2labels}}
                img = Image.open(
                    f"{DATA_DIR}/train/{images_ids[i]}.jpg").convert("RGB")
                img = wandb.Image(
                    img, 'RGB', caption=f"{images_ids[i]}", boxes=boxes_dict)
                wandb.log({f"{images_ids[i]}": img})
            batch_precision /= len(predictions)
            wandb.log({"batch_precision": batch_precision})


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Random seeds
    seed = wandb.config.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data loaders
    frac = wandb.config.frac
    batch_size = wandb.config.batch_size
    num_workers = wandb.config.num_workers

    data = dataPreprocessing.dataPreprocess(DATA_DIR)
    train_ids, val_ids = data.random_split_dataset(frac)

    tsfm = utils.get_transforms(
        wandb.config.transforms)

    # tsfm = albumentations.Compose([
    #     # albumentations.Resize(800, 800),
    #     albumentations.Flip(0.5),
    #     ToTensorV2(p=1.0)
    # ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    val_tsfm = albumentations.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    train_dataset = wheatDataloader.WheatDataset(
        train_ids, DATA_DIR, transforms=tsfm)
    val_dataset = wheatDataloader.WheatDataset(
        val_ids, DATA_DIR, transforms=val_tsfm)
    eval_dataset = wheatDataloader.WheatDatasetTest(test_dir='test')

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1)

    # Model
    optimizer = wandb.config.optimizer
    learning_rate = wandb.config.learning_rate
    lr_step = wandb.config.lr_step
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes=2)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer == 'Adam':
        optimizer = utils.get_optimizer(optimizer)(params, learning_rate)
    if optimizer == 'SGD':
        optimizer = utils.get_optimizer(optimizer)(
            params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step)

    # Loop
    epochs = wandb.config.epochs
    for epoch in range(1, epochs+1):
        train(model, train_dataloader, optimizer, epoch, device, lr_scheduler)
        test(model, val_dataloader, epoch, device)
        # evaluation(model, eval_dataloader)

    torch.save(model, 'model.pth')


if __name__ == "__main__":
    main()
