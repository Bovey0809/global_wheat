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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from matplotlib import pyplot as plt

import dataPreprocessing
import wheatDataloader
from wheatDataloader import collate_fn
import wandb
wandb.init(project="global_wheat")


DATA_DIR = 'data'
# torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DataPrerpocessing
data = dataPreprocessing.dataPreprocess(DATA_DIR)
train_ids, val_ids = data.random_split_dataset()

# Dataset
tsfm = transforms.Compose([
    transforms.ToTensor()
])
target_tsfm = torch.ones_like
dataset = wheatDataloader.WheatDataset(train_ids, DATA_DIR, transforms=tsfm)


dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=16, num_workers=8, collate_fn=collate_fn)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True, progress=True, pretrained_backbone=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

wandb.watch(model)

# model = nn.DataParallel(model)
model.to(device=device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-3)

# loop over the dataset multiple times

# For saving best model
best_loss = float('inf')

# For early stop
steps = 3
previous_loss = []

for epoch in range(2):
    for iteration, data in enumerate(dataloader, 0):
        images, targets, image_ids = data
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
        if iteration % 50:
            print(f"Iteration {iteration} Loss: {loss.mean().item()}")

torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

print('Finished Training')
