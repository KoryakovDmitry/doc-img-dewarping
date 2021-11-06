import cv2
from glob import glob
from PIL import Image
import os
import os.path as osp
import numpy as np
import json

base = os.getcwd()
imgs = glob(osp.join(base, "test_imgs/*"))
anns_dir = osp.join(base, "test_anns/")

for im_path in imgs:
    im_fn = osp.basename(im_path)
    img = cv2.imread(im_path)

    with open(osp.join(anns_dir, im_fn.replace(".jpg", ".json"))) as f:
        anns = json.load(f)

    points = anns["shapes"][0]["points"]
    pts = np.array(points).astype(int)
    img = cv2.polylines(img, [pts], True, color=(0, 0, 255), thickness=2)
    for p in pts:
        img = cv2.circle(img, (p[0], p[1]), radius=10, color=(255, 255, 0), thickness=-1)

    x1, y1, w, h = cv2.boundingRect(pts)
    x2, y2 = x1 + w, y1 + h
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)

    Image.fromarray(img[:, :, ::-1]).show()


