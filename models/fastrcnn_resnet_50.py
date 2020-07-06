import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class fastercnn(torch.nn.Module):
