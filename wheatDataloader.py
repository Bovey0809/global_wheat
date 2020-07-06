import glob
import os
import random

import cv2
import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms

import dataPreprocessing


class WheatDataset(torch.utils.data.Dataset):
    """Global wheat dataset for dataloader.

    Attributes:
        image_ids: list of file names, ['fe133ccb4', ...]
        image_dir: data path, data/train/ or data/test/
        transforms: preprocessing for images from torchvision.
        target_transforms: transform target

    Returns:
        image: PIL RGB format.
        bboxes: [N, 4] x1, y1, x2, y2
        labels: [N, ] [1, ..., 1]
        areas: [N, ] areas float
    """

    def __init__(self, image_ids, image_dir, transforms=None, target_transforms=None):
        super(WheatDataset, self).__init__()
        self.image_ids = image_ids
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.img_dir = image_dir
        meta_data = dataPreprocessing.dataPreprocess(image_dir).data_df
        meta_data['x2'] = meta_data.x + meta_data.w
        meta_data['y2'] = meta_data.y + meta_data.h
        meta_data['area'] = meta_data.w * meta_data.h
        self.dataframe = meta_data.groupby('image_id')

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        data = self.dataframe.get_group(
            image_id).loc[:, ['x', 'y', 'x2', 'y2', 'source', 'area']].values
        bboxes, labels, areas = data[:, :4], data[:, -2], data[:, -1]

        bboxes = torch.from_numpy(bboxes.astype(np.float32))
        labels = torch.ones(labels.shape, dtype=torch.int64)

        # image
        img_path = os.path.join(self.img_dir, 'train', image_id + '.jpg')
        image = Image.open(img_path).convert('RGB')
        # The image is in HWC, we need to convert to CHW.
        if self.transforms:
            image = self.transforms(image)
        if self.target_transforms:
            labels = self.target_transforms(labels)
        return image, bboxes, labels, image_id

    def __len__(self):
        return len(self.image_ids)


class WheatDatasetTest(torch.utils.data.Dataset):
    """Some Information about WheatDatasetTest  """

    def __init__(self, test_dir):
        super(WheatDatasetTest, self).__init__()
        # glob all the test data
        self.image_ids = glob.glob(os.path.join(test_dir, "*.jpg"))

    def __getitem__(self, index):
        img = PIL.Image.open(self.image_ids[index]).convert('RGB')
        img = torchvision.transforms.ToTensor()(img)
        return img, self.image_ids[index]

    def __len__(self):
        return len(self.image_ids)


def collate_fn(batch):
    images, bboxes, labels, imageid = tuple(zip(*batch))
    targets = []
    for box, label in zip(bboxes, labels):
        targets.append({'boxes': box, 'labels': label})
    return torch.stack(images), targets, imageid


if __name__ == "__main__":
    data_path = 'data'
    train_ids, val_ids = dataPreprocessing.dataPreprocess(
        data_path).random_split_dataset()

    # Test dataloader
    tsfm = torchvision.transforms.Compose([transforms.ToTensor()])

    dataset = WheatDataset(
        val_ids, data_path, transforms=tsfm)
    image, bbox, label, imageid = dataset[random.randint(
        0, len(dataset))]
