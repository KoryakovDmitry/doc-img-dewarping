import cv2
import math
from copy import deepcopy
import numpy as np
from sympy.geometry import Line, Point2D
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from inference_segm import (
    SegmInference,
    plot_border_corrected as plot_border_corrected_,
)
from shapely.geometry import Polygon


def plot_border_corrected(img_plot, destination_pts_, points_, add_x=0, add_y=0):
    destination_pts = deepcopy(destination_pts_)
    points = deepcopy(points_)
    for side in destination_pts:
        for p_i in range(len(destination_pts[side])):
            destination_pts[side][p_i][0] += add_x
            destination_pts[side][p_i][1] += add_y

    destination_pts_list = [i for l in destination_pts for i in destination_pts[l]]
    for side in destination_pts:
        for dst_pts in destination_pts[side]:
            p = np.array(dst_pts).astype(int)
            img_plot = cv2.circle(
                img_plot,
                (p[0], p[1]),
                radius=10,
                color=(0, 255, 0),
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

    for pt_src, pt_dst in zip(points, destination_pts_list):
        pt_src = np.array(pt_src).astype(int)
        pt_dst = np.array(pt_dst).astype(int)

        img_plot = cv2.line(
            img_plot,
            tuple(pt_src),
            tuple(pt_dst),
            (0, 0, 255),
            thickness=2,
        )

    Image.fromarray(img_plot[:, :, ::-1]).show()

    # save_to = f"/Users/dmitry/Initflow/doc-img-dewarping/dewarp_homo/"
    # name = str(points_[0]).replace(",", "").replace(".", "").replace(" ", "").replace("[", "").replace("]", "")+".jpg"
    # cv2.imwrite(osp.join(save_to, name), img_plot)
    return img_plot


def order_points_clockwise(pts):
    rect = np.zeros((4, 2))
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_destination_points(points):

    whs = {"w": [], "h": []}
    for pt_i_next in range(1, len(points)):
        pt_i = pt_i_next - 1
        pt = points[pt_i]
        pt_next = points[pt_i_next]

        hypot = np.hypot(
            abs(pt[0] - pt_next[0]),
            abs(pt[1] - pt_next[1]),
        )

        if pt_i_next == 1:
            whs["w"].append(hypot)
        elif pt_i_next == 2:
            whs["h"].append(hypot)
        elif pt_i_next == 3:
            whs["w"].append(hypot)

    pt = points[pt_i_next]
    pt_next = points[0]

    hypot = np.hypot(
        abs(pt[0] - pt_next[0]),
        abs(pt[1] - pt_next[1]),
    )
    whs["h"].append(hypot)

    w, h = max(whs["w"]), max(whs["h"])

    destination_pts = [[0, 0], [w, 0], [w, h], [0, h]]

    return destination_pts, h, w


def unwarp(img, src, dst):
    """

    Args:
        img: np.array
        src: list
        dst: list

    Returns:
        un_warped: np.array

    """
    h, w = img.shape[:2]
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=9.0)
    # H = cv2.getPerspectiveTransform(src, dst)
    un_warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)
    return un_warped


def dewarp(image, pts_src):
    pts_src = np.array(Polygon(pts_src).minimum_rotated_rectangle.exterior.coords)[:-1]
    pts_src = order_points_clockwise(pts_src)

    # debug
    img_plot = image.copy()
    for n, p in enumerate(pts_src):
        p = np.array(p).astype(int)
        img_plot = cv2.circle(
            img_plot, (p[0], p[1]), radius=10, color=(255, 255, 0), thickness=-1
        )
        img_plot = cv2.putText(
            img_plot,
            str(n),
            (p[0] + 50, p[1] + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    # Image.fromarray(img_plot[:, :, ::-1]).show()

    destination_pts, h, w = get_destination_points(points=pts_src)
    # destination_pts = order_points_clockwise(destination_pts)
    for n, p in enumerate(destination_pts):
        p = np.array(p).astype(int)
        img_plot = cv2.circle(
            img_plot, (p[0], p[1]), radius=10, color=(0, 255, 255), thickness=-1
        )
        img_plot = cv2.putText(
            img_plot,
            str(n),
            (p[0] + 50, p[1] + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    Image.fromarray(img_plot[:, :, ::-1]).show()
    #
    h, w = int(np.round(h)), int(np.round(w))
    un_warped = unwarp(image, np.float32(pts_src), np.float32(destination_pts))
    cropped = un_warped[0:h, 0:w]
    return un_warped, cropped


def group_points(points, tol):
    groups = []
    while points:
        far_points = []
        ref = points.pop()
        groups.append([ref])
        for point in points:
            d = get_distance(ref, point)
            if d < tol:
                groups[-1].append(point)
            else:
                far_points.append(point)

        points = far_points

    # perform average operation on each group
    return [list(np.mean(x, axis=0).astype(int)) for x in groups]


def get_distance(ref, point):
    # print('ref: {} , point: {}'.format(ref, point))
    x1, y1 = ref
    x2, y2 = point
    return math.hypot(x2 - x1, y2 - y1)


if __name__ == "__main__":
    import cv2
    from PIL import Image
    import numpy as np
    import json
    import os
    import os.path as osp
    from glob import glob

    segm = SegmInference()
    base = os.getcwd()
    imgs = glob(osp.join(base, "test_imgs/*"))
    # anns_dir = osp.join(base, "test_anns_4_pts/")
    anns_dir = osp.join(base, "test_anns/")
    for im_path in imgs:
        im_fn = osp.basename(im_path)
        img = cv2.imread(im_path)

        # with open(osp.join(anns_dir, im_fn.replace(".jpg", ".json"))) as f:
        #     anns = json.load(f)

        points = segm.inference(img)
        h = plot_border_corrected_(img.copy(), points)

        tol_real = 2.2058823529411766
        tol = np.hypot(img.shape[0], img.shape[1]) * 0.01 * tol_real
        points_grouped = group_points(points, tol=tol)
        h = plot_border_corrected_(img.copy(), points_grouped)

        img_plot = img.copy()
        for n, p in enumerate(points_grouped):
            p = np.array(p).astype(int)
            img_plot = cv2.circle(
                img_plot, (p[0], p[1]), radius=10, color=(255, 255, 0), thickness=-1
            )
            img_plot = cv2.putText(
                img_plot,
                str(n),
                (p[0] + 50, p[1] + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        un_warped, cropped = dewarp(image=img, pts_src=points_grouped)
        # Image.fromarray(un_warped[:, :, ::-1]).show()
        Image.fromarray(cropped[:, :, ::-1]).show()
