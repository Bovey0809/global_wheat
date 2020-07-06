# %%
# randomly show imgs
from sklearn.model_selection import train_test_split
import random
import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib import pyplot as plt
from PIL import Image
import copy


class dataPreprocess(object):
    """Convert train.csv into Train and Validation parts.

    read csv file → convert bbox → drop columns → train and val image ids

    Attributes:
        train_csv_path: train.csv dir

    """

    def __init__(self, data_path):
        """Read data root path.

        Args:
            data_path: data which contains images and train.csv.
        """
        self.data_path = data_path
        self.data_csv_path = os.path.join(data_path, 'train.csv')
        self.data_df = pd.read_csv(self.data_csv_path)

        # conver bbox from string into xmin, ymin, w, h, (top left point, w, h)
        bboxs = np.stack(self.data_df.bbox.apply(
            lambda x: np.fromstring(x[1:-1], sep=',')))
        for i, column in enumerate(['x', 'y', 'w', 'h']):
            self.data_df[column] = bboxs[:, i]
        self.data_df.drop(columns=['bbox', 'width', 'height'], inplace=True)
        self.img_ids = self.data_df.image_id.unique()

    def random_split_dataset(self, frac=0.8):
        """Random split dataset into train and validation based on frac.

        Args:
            frac: frac as train dataset, (1-frac) as validation set.

        Returns:
            return list(train_ids), list(val_ids)
        """
        # In case you need the original img_ids, we copy the imgs and shuffle.
        img_ids = copy.deepcopy(self.img_ids)
        random.shuffle(img_ids)
        train_index = int(frac * len(img_ids))
        train_ids = img_ids[:train_index]
        val_ids = img_ids[train_index:]
        return train_ids, val_ids

    def __str__(self):
        return self.data_df.describe().__str__() + self.data_df.info().__str__()


def draw_rectangle(img_path, x, y, w, h):
    img = Image.open(img_path)
    cv2.rectangle(img, (x, y), (x + w, y + h), 255, 3)
    return img


def draw_multirectangles(boxes, imgpath):
    img = plt.imread(imgpath)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), 255, 3)
    return img


if __name__ == "__main__":
    train_csv_path = 'data'
    # Test init
    test_preprocess = dataPreprocess(train_csv_path)
    print(test_preprocess)
    # Test split
    train_ids, val_ids = test_preprocess.random_split_dataset()
    print(len(train_ids), len(val_ids))
