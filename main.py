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


np.random.seed(0)
torch.manual_seed(0)

hyperparameter_defaults = dict(
    batch_size=16,
    learning_rate=0.001,
    epochs=2,
    opt='adam'  # optimizer
)
wandb.init(project="global_wheat", config=hyperparameter_defaults)
DATA_DIR = 'data'
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" DATA """
# DataPrerpocessing
data = dataPreprocessing.dataPreprocess(DATA_DIR)
train_ids, val_ids = data.random_split_dataset(frac=0.8)

# Dataset
tsfm = transforms.Compose([
    transforms.ToTensor()
])
target_tsfm = torch.ones_like
dataset = wheatDataloader.WheatDataset(train_ids, DATA_DIR, transforms=tsfm)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=hyperparameter_defaults['batch_size'], num_workers=8, collate_fn=collate_fn)

val_dataset = wheatDataloader.WheatDataset(val_ids, DATA_DIR, transforms=tsfm)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=16, num_workers=8, collate_fn=collate_fn
)

""" MODEL """
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True, progress=True, pretrained_backbone=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
# model = nn.DataParallel(model)
model.to(device=device)

params = [p for p in model.parameters() if p.requires_grad]
if hyperparameter_defaults['opt'] == 'adam':
    optimizer = torch.optim.Adam(
        params, lr=hyperparameter_defaults['learning_rate'])
else:
    raise Exception("You have to choose one optimizer.")

# loop over the dataset multiple times

# For saving best model
best_loss = float('inf')

# For early stop
steps = 3
previous_loss = []

for epoch in range(hyperparameter_defaults['epochs']):
    for iteration, (images, targets, image_ids) in enumerate(dataloader, 0):
        model.train(True)
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
        wandb.log({'Training_loss': loss.item(),
                   'epoch': epoch, 'iter': iteration})
        if iteration % 50 == 0:
            model.eval()
            # TODO Implement mAP for validation set.
            # TODO: Implement mAP for training set.
            # Calculate precision for training.
            predictions = model(images)
            for idx, prediction in enumerate(predictions):
                boxes, labels, scores = prediction.values()
                if boxes.shape[0] == 0:
                    continue
                else:
                    gts = targets[idx]['boxes']
                    # The scores is already sorted by the model.
                    image_precision = calculate_image_precision(
                        gts, boxes)
                    print(image_precision)
torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

print('Finished Training')
