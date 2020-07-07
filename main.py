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

hyperparameter_defaults = dict(
    batch_size=16,
    learning_rate=0.001,
    epochs=2,
    opt='adam',  # optimizer
    frac=0.8,  # train / (train + val)
    seed=0,
    num_workers=8,
    # To tensor is added to the last as default.
    transforms=['RandomHorizontalFlip',
                'RandomGrayscale', 'RandomRotation', 'RandomVerticalFlip'],
    optimizer='Adam'  # SGD is also support
)
wandb.init(project="global_wheat", config=hyperparameter_defaults)
DATA_DIR = 'data'
# torch.backends.cudnn.benchmark = True


def train(model, dataloader, optimizer, epoch, device):
    model.train(True)
    for iteration, (images, targets, images_ids) in enumerate(dataloader, 0):
        images = images.to(device)
        targets = [{'boxes': i['boxes'].to(
            device), 'labels':i['labels'].to(device)} for i in targets]
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()
        # Log Training info
        # loss
        for k, v in loss_dict.items():
            wandb.log({'_'.join(['Train', k]): v})


def test(model, dataloder):
    model.eval()
    test_loss = 0
    precision = 0
    best_loss = 100
    with torch.no_grad():
        for iteration, (images, targets, images_ids) in enumerate(dataloder, 0):
            images = images.to(device)
            targets = [{'boxes': i['boxes'].to(
                device), 'labels':i['labels'].to(device)} for i in targets]
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            # loss
            for k, v in loss_dict.items():
                wandb.log({'_'.join(['Validation', k]): v})


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Random seeds
    seed = hyperparameter_defaults['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data loaders
    frac = hyperparameter_defaults['frac']
    batch_size = hyperparameter_defaults['batch_size']
    num_workers = hyperparameter_defaults['num_workers']

    data = dataPreprocessing.dataPreprocess(DATA_DIR)
    train_ids, val_ids = data.random_split_dataset(frac)

    tsfm = utils.get_transforms(
        hyperparameter_defaults['transforms'])

    train_dataset = wheatDataloader.WheatDataset(
        train_ids, DATA_DIR, transforms=tsfm)
    val_dataset = wheatDataloader.WheatDataset(
        val_ids, DATA_DIR, transforms=tsfm)
    eval_dataset = wheatDataloader.WheatDatasetTest(test_dir='test')

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1)

    # Model
    optimizer = hyperparameter_defaults['optimizer']
    learning_rate = hyperparameter_defaults['learning_rate']

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True, pretrained_backbone=True
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes=2)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = utils.get_optimizer(optimizer)(params, lr=learning_rate)

    # Loop
    epochs = hyperparameter_defaults['epochs']
    for epoch in range(1, epochs+1):
        train(model, train_dataloader, optimizer, epoch, device)
        test(model, val_dataloader, device)
        evaluation(model, eval_dataloader)


if __name__ == "__main__":
    main()
