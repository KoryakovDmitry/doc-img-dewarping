from detectron2.utils.visualizer import ColorMode
from copy import deepcopy
from PIL import Image

# import some common libraries
import cv2
from glob import glob

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from detectron2.utils.visualizer import Visualizer
import os
import os.path as osp

from detectron2.config import get_cfg

import numpy as np
from skimage import measure
from rdp import rdp

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:


def plot_border_corrected(img_plot, destination_pts_, add_x=0, add_y=0):
    destination_pts_list = deepcopy([list(i) for i in destination_pts_])
    for p_i in range(len(destination_pts_list)):
        destination_pts_list[p_i][0] += add_x
        destination_pts_list[p_i][1] += add_y

    for dst_pts in destination_pts_list:
        p = np.array(dst_pts).astype(int)
        img_plot = cv2.circle(
            img_plot,
            (p[0], p[1]),
            radius=10,
            color=(0, 0, 255),
            thickness=-1,
        )

    for dst_pts_i in range(0, len(destination_pts_list) - 1):
        dst_pts_i_next = dst_pts_i + 1
        dst_pts_next = np.array(destination_pts_list[dst_pts_i_next]).astype(int)
        dst_pts = np.array(destination_pts_list[dst_pts_i]).astype(int)

        img_plot = cv2.line(
            img_plot,
            tuple(dst_pts),
            tuple(dst_pts_next),
            (255, 0, 0),
            thickness=3,
        )

    img_plot = cv2.line(
        img_plot,
        tuple(np.array(destination_pts_list[dst_pts_i_next]).astype(int)),
        tuple(np.array(destination_pts_list[0]).astype(int)),
        (255, 0, 0),
        thickness=3,
    )
    Image.fromarray(img_plot[:, :, ::-1]).show()

    # save_to = f"/Users/dmitry/Initflow/doc-img-dewarping/dewarp_homo/"
    # name = str(points_[0]).replace(",", "").replace(".", "").replace(" ", "").replace("[", "").replace("]", "")+".jpg"
    # cv2.imwrite(osp.join(save_to, name), img_plot)
    return img_plot


class SegmInference:
    def __init__(self):
        cfg = get_cfg()
        cfg.OUTPUT_DIR = osp.join(os.getcwd(), "dewarp_homo/weights")
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.DATASETS.TRAIN = ("train",)
        cfg.DATASETS.TEST = ("val",)
        cfg.TEST.EVAL_PERIOD = 100
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = os.path.join(
            cfg.OUTPUT_DIR, "model_final.pth"
        )  # path to the model we just trained
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 4000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        cfg.SOLVER.STEPS = []  # do not  decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            128  # faster, and good enough for this toy dataset (default: 512)
        )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
        # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
        cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(cfg)

    def inference(self, im, eps=0.999999999999):
        outputs = self.predictor(im)
        mask = np.transpose(
            outputs["instances"]._fields["pred_masks"].numpy()[0], (1, 0)
        )
        mask = mask.astype(np.uint8)
        contours = measure.find_contours(mask, 0.5)

        polygons = []
        for object in contours:
            coords = []

            for point in object:
                coords.append([int(point[0]), int(point[1])])

            polygons.append(coords)

        points = rdp(np.array(polygons[0]), epsilon=eps)

        points_new = []
        for p in points:
            p = p.tolist()
            if p not in points_new:
                points_new.append(p)

        return points_new


if __name__ == "__main__":
    imgs_path = glob("/Users/dmitry/Initflow/doc-img-dewarping/test_imgs/*")
    for i_p in imgs_path:
        im = cv2.imread(i_p)
        from datetime import datetime

        cfg = get_cfg()
        cfg.OUTPUT_DIR = osp.join(os.getcwd(), "weights")
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.DATASETS.TRAIN = ("train",)
        cfg.DATASETS.TEST = ("val",)
        cfg.TEST.EVAL_PERIOD = 100
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = os.path.join(
            cfg.OUTPUT_DIR, "model_final.pth"
        )  # path to the model we just trained
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 4000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        cfg.SOLVER.STEPS = []  # do not  decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            128  # faster, and good enough for this toy dataset (default: 512)
        )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
        # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
        cfg.MODEL.DEVICE = "cpu"
        predictor = DefaultPredictor(cfg)

        c = datetime.now()
        outputs = predictor(im)
        mask = np.transpose(
            outputs["instances"]._fields["pred_masks"].numpy()[0], (1, 0)
        )
        mask = mask.astype(np.uint8)
        contours = measure.find_contours(mask, 0.5)

        polygons = []
        for object in contours:
            coords = []

            for point in object:
                coords.append([int(point[0]), int(point[1])])

            polygons.append(coords)

        polygons = rdp(np.array(polygons[0]), epsilon=0.999999999999)

        im_plot = im.copy()
        im_plot = plot_border_corrected(im_plot, destination_pts_=polygons)

        print(datetime.now() - c)
        v = Visualizer(
            im[:, :, ::-1],
            scale=0.5,
            instance_mode=ColorMode.IMAGE_BW,
            # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = out.get_image()

        Image.fromarray(img).show()

        k = None
