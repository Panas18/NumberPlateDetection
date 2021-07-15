import cv2
import torch
import torch.optim as optim
from model import Yolov1
from utils import (
        non_max_suppression,
        mean_average_precision,
        intersection_over_union,
        cellboxes_to_boxes,
        get_bboxes,
        plot_image,
        save_checkpoint,
        load_checkpoint,
        )

# Hyperparameters etc.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOAD_MODEL = True
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
test_file = 'test.csv'
LEARNING_RATE = 2e-1
WEIGHT_DECAY = 0


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=1).to(DEVICE)
    optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
    load_checkpoint(torch.load(LOAD_MODEL_FILE,
        map_location=torch.device('cpu')), model, optimizer)
    img_dir = 'data/new_img/26.jpg'
    image = cv2.imread(img_dir, 0)
    print(image.shape)
    image = cv2.resize(image, (448, 448))
    image = torch.tensor(image).float()
    image = torch.reshape(image, (1, 1, 448, 448))
    pred = model(image)
    print(pred.shape)
    bboxes = cellboxes_to_boxes(pred)
    for box in bboxes[0]:
        print(box[1])
    bboxes = non_max_suppression(
            bboxes[0], iou_threshold = 0.5, threshold = 0.4, box_format = 'midpoint')
    print(bboxes)
if __name__ == "__main__":
    main()
