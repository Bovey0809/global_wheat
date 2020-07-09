import wheatDataloader
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random
import re
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import dataPreprocessing

dir_input = 'data'
dir_test = f'{dir_input}/test'
dir_train = f'{dir_input}/train'

train_df = pd.read_csv(f'{dir_input}/train.csv')
train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1


def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r


# 取出来bbox的坐标，分别用x,y,w,h表示
train_df[['x', 'y', 'w', 'h']] = np.stack(
    train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)  # 去掉bbox那个列表，因为已经用x,y,w,h表示了
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)

image_ids = train_df['image_id'].unique()
vaild_ids = image_ids[-655:]  # 把图片分为训练和验证集合
train_ids = image_ids[:-655]
vaild_df = train_df[train_df['image_id'].isin(vaild_ids)]
train_df = train_df[train_df['image_id'].isin(train_ids)]


class WheatDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(
            f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)  # 读入图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(
            np.float32)  # 处理图片变成改变颜色
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0]+boxes[:, 2]  # 变成左上，右下的方式
        boxes[:, 3] = boxes[:, 1]+boxes[:, 3]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:  # 数组增强
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.stack(
                tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self):
        return self.image_ids.shape[0]


def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


frcnn_dataset = WheatDataset(train_df, dir_train, get_train_transform())


"""My dataset"""
data_path = 'data'
train_ids, val_ids = dataPreprocessing.dataPreprocess(
    data_path).random_split_dataset(1.0)

tsfm = A.Compose([
    A.Resize(800, 800),
    A.Flip(0.5),
    ToTensorV2(p=1.0)
], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

dataset = wheatDataloader.WheatDataset(
    train_ids, data_path, transforms=get_train_transform())
# image, bbox, label, imageid = dataset[random.randint(
#     0, len(dataset))]
# print(image, bbox, label)


for his_data, my_data in zip(frcnn_dataset, dataset):
    print(his_data)
    print(my_data)
