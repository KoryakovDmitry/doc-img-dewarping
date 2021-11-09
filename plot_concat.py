import os
import os.path as osp
from glob import glob
import cv2


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [
        cv2.resize(
            im,
            (int(im.shape[1] * h_min / im.shape[0]), h_min),
            interpolation=interpolation,
        )
        for im in im_list
    ]
    return cv2.hconcat(im_list_resize)


base = os.getcwd()
dir_orig, _, orig = list(os.walk(osp.join(base, "output_orig")))[0]
dir_poly, _, poly = list(os.walk(osp.join(base, "output_poly")))[0]
dir_output, _, output = list(os.walk(osp.join(base, "output")))[0]

intr = set(orig) & set(poly) & set(output)

to = osp.join(base, "output_concat")
os.makedirs(to, exist_ok=True)


for i in intr:
    im1 = cv2.imread(osp.join(dir_orig, i))
    im2 = cv2.imread(osp.join(dir_poly, i))
    im3 = cv2.imread(osp.join(dir_output, i))
    im_h_resize = hconcat_resize_min([im1, im2, im3])
    cv2.imwrite(osp.join(to, i), im_h_resize)
