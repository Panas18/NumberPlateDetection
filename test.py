import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np
import torch.optim as optim
from model import Yolov1
from loss import YoloLoss
from dataset import VOCDataset
from torch.utils.data import DataLoader
from utils import (
        non_max_suppression,
        cellboxes_to_boxes,
        get_bboxes,
        load_checkpoint,
        mean_average_precision
        )
import torchvision.transforms as transforms
from train import Compose

# Hyperparameters etc.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = 'data/labels'
LEARNING_RATE = 2e-1
WEIGHT_DECAY = 0
BATCH_SIZE = 1 
NUM_WORKERS = 2
PIN_MEMORY = True
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])


def plot_image(image, bboxes, loss):
    image = torch.reshape(image, (448, 448)).numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    if len(bboxes) == 0:
        plt.show()
    else:
        for box in bboxes:
            prob, x, y, w, h = box[1], box[2], box[3], box[4], box[5]
            loss = loss.detach().numpy()
            loss = np.around(loss,2)
            xmin, ymin = x - w/2, y - h/2
            rect = patches.Rectangle(
                    (xmin * 448, ymin * 448),
                    w * 448,
                    h * 448,
                    linewidth=1,
                    edgecolor='r',
                    facecolor='none',
                    label=str(loss)
                    )
            plt.text(xmin * 440, ymin * 440, str(loss), color='r')
            ax.add_patch(rect)
        plt.show()


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=1).to(DEVICE)
    optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
    load_checkpoint(torch.load(LOAD_MODEL_FILE,
        map_location=torch.device('cpu')), model, optimizer)

    test_dataset = VOCDataset(
            "data/test.csv",
            transform=transform,
            img_dir=IMG_DIR,
            label_dir=LABEL_DIR,
            )
    loss_fun = YoloLoss()
    test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            shuffle=True,
            drop_last=False,
            )
    avg_loss = []
    pred_boxes, target_boxes = get_bboxes(
            test_loader, model, iou_threshold=0.5, threshold=0.5
            )
    mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fun(out, y)
        print(f"Loss: {loss.item()}")
        avg_loss.append(loss.item())
        bboxes = cellboxes_to_boxes(out)
        bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5,
                threshold=0.4, box_format='midpoint')
        plot_image(x, bboxes, loss)
    print(f"\nAverage Training Loss: {sum(avg_loss)/len(avg_loss)}")
    print(f"Test mAP : {mean_avg_prec}")


if __name__ == "__main__":
    main()
