# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd


# %%
# 因为数据集比较小， 只有600M，为了性能，可以一次读取所有的图片到内存上面。


# %%
data_path = './data/'


# %%
import os
train_data_path = os.path.join(data_path, 'train')


# %%
# 对于所有的类别做可视化
import pandas as pd


# %%
train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))


# %%
train_df.describe()


# %%
train_df.info()


# %%
# 从上面的信息可以看出来，没有空值，然后下一步，理解这几列的含义。


# %%
train_df.sample()


# %%
os.path.exists(os.path.join(train_data_path, '85189f1bf.jpg'))


# %%
get_ipython().system('which python')


# %%
import matplotlib
import PIL


# %%
# randomly pick up a image from train floder
img_id = train_df.sample()['image_id'] + '.jpg'


# %%
import matplotlib.pyplot as plt


# %%
img_id.values[0]


# %%
img = plt.imread(os.path.join(train_data_path, img_id.values[0]))  


# %%
plt.imshow(img)


# %%
img.shape


# %%
# width 和 height 就是图片的长和宽


# %%
train_df.bbox.sample(5)


# %%
random_img = train_df.sample()


# %%
path = os.path.join(train_data_path, random_img.image_id.values[0]+'.jpg')


# %%
plt.imshow(plt.imread(path))


# %%
import cv2


# %%


