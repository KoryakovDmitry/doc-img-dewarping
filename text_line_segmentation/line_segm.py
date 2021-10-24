import os
import cv2
import numpy as np
from glob import glob
import os.path as osp
from tqdm import tqdm
from PIL import Image


path = "../test_imgs/33.jpg"
# Read gray image
img = cv2.imread(path, 0)
img_denoised = cv2.fastNlMeansDenoising(img)

# Create default parametrization LSD
lsd = cv2.createLineSegmentDetector(0)

# Detect lines in the image
lines = lsd.detect(img_denoised)[0]  # Position 0 of the returned tuple are the detected lines

# Draw detected lines in the image
# drawn_img = lsd.drawSegments(img, lines)
lengs = np.hypot(abs(lines[:, 0, 0] - lines[:, 0, 2]), abs(lines[:, 0, 1] - lines[:, 0, 3]))
mean_leng = lengs.mean()

drawn_img = img_denoised.copy()
for i_ in lines:
    i = i_[0]
    start_point = (int(i[0]), int(i[1]))
    end_point = (int(i[2]), int(i[3]))
    leng = np.hypot(abs(start_point[0] - end_point[0]), abs(start_point[1] - end_point[1]))
    if mean_leng < leng:
        drawn_img = cv2.line(drawn_img, start_point, end_point, (0, 0, 255), 2)


Image.fromarray(drawn_img).show()
