import cv2
import os
import numpy as np
image_dir = 'ashutoshimg'
out_dir = 'new_gray'

def main(image_dir):
    images =  os.listdir(image_dir)
    for image in images:
        img_dir = os.path.join(image_dir, image)
        gray_dir = os.path.join(out_dir, image)
        img = cv2.imread(img_dir)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray_img.shape[0],gray_img.shape[1]
        gray_img = gray_img.reshape(height, width,1)
        cv2.imwrite(gray_dir, gray_img)
        
if __name__ == "__main__":
    main(image_dir)
    print('complete')

