import wandb
from dataPreprocessing import dataPreprocess
from utils import create_box_data, draw_image
from wheatDataloader import WheatDataset

wandb.init(project='global_wheat', tags=["image test"])

# TODO: Draw ground truth images to W&B.

# Get the data first
data = dataPreprocess('data')
_, data_idx = data.random_split_dataset(0.999)

datasets = WheatDataset(data_idx, 'data')
for dataset in datasets:
    image, boxes, labels, image_id = dataset
    box_data = create_box_data(boxes)
    wandb_img = draw_image(image, box_data)
    wandb.log({f'{image_id}': wandb_img})
