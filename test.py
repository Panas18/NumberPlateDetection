import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import random
import torch.optim as optim
import pandas as pd
from model import Yolov1
from utils import (
        non_max_suppression,
        mean_average_precision,
        intersection_over_union,
        cellboxes_to_boxes,
        get_bboxes,
        save_checkpoint,
        load_checkpoint,
        )

# Hyperparameters etc.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOAD_MODEL = True
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
TEST_CSV = pd.read_csv('data/test.csv')
LEARNING_RATE = 2e-1
WEIGHT_DECAY = 0
TEST_SIZE = 47


def preprocess_image(img_dir):
    image = cv2.imread(img_dir, 0)
    image = cv2.resize(image, (448, 448))
    image = torch.tensor(image).float()
    image = torch.reshape(image, (1, 1, 448, 448))
    return image


def plot_image(image, bboxes):
    image = torch.reshape(image, (448, 448)).numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap = 'gray')
    if len(bboxes) == 0:
        plt.show()
    else:
        for box in bboxes:
            prob, x, y, w, h = box[1], box[2], box[3], box[4], box[5]
            prob = round(prob, 2)
            xmin, ymin = x - w/2, y - h/2
            #xmax, ymax = x + w/2, y + h/2
            rect = patches.Rectangle(
                    (xmin * 448, ymin * 448),
                    w * 448,
                    h * 448,
                    linewidth = 1,
                    edgecolor = 'r',
                    facecolor = 'none',
                    label = str(prob)
                    )
            plt.text(xmin* 440, ymin*440,str(prob), color = 'r')
            ax.add_patch(rect)
        plt.show()


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=1).to(DEVICE)
    optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
    load_checkpoint(torch.load(LOAD_MODEL_FILE,
        map_location=torch.device('cpu')), model, optimizer)

    for i in range(TEST_SIZE):
        index = random.randint(0, TEST_SIZE -1)
        test_img= TEST_CSV.iloc[index, 0]
        img_dir = os.path.join(IMG_DIR,test_img)
        image = preprocess_image(img_dir)
        pred = model(image)
        bboxes = cellboxes_to_boxes(pred)
        bboxes = non_max_suppression(
                bboxes[0], iou_threshold = 0.5, threshold = 0.4, box_format = 'midpoint')
        plot_image(image, bboxes)


if __name__ == "__main__":
    main()
