import math
from copy import deepcopy
import numpy as np
from sympy.geometry import Line, Point2D

from .segmentation import (
# from segmentation import (
    Segmentator,
    plot_border_corrected as plot_border_corrected_,
)
import cv2
from PIL import Image
from itertools import combinations
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

    # save_to = f"/Users/dmitry/Initflow/doc-img-dewarping/homo_warp_trans/"
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


def dewarp(image, pts_src, debug_plot):
    # pts_src = order_points_clockwise(pts_src)

    # debug
    if debug_plot:
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
    if debug_plot:
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
        # Image.fromarray(img_plot[:, :, ::-1]).show()
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
    return np.array([list(np.mean(x, axis=0).astype(int)) for x in groups])


def get_distance(ref, point):
    x1, y1 = ref
    x2, y2 = point
    return math.hypot(x2 - x1, y2 - y1)


def get_angle_by_idx(angles, idx):
    for a in angles:
        if idx == a[0][1]:
            return a


def get_max_poly4(idxs, points):
    pi1, pi2, pi3, pi4 = idxs
    p1, p2, p3, p4 = points[pi1], points[pi2], points[pi3], points[pi4]
    p = Polygon([p1, p2, p3, p4])
    return p.area


def polyg(start, end, l):
    i = start
    pts = []
    if end == 0:
        end = l
    while True:
        if i == end - 1:
            break
        pts.append(i)

        if i == 0:
            i = l - 1
        else:
            i -= 1
    return pts


def below_above_line_filter(line, pts, side):
    new_points = []
    line = np.array(line)
    v1 = (line[1][0] - line[0][0], line[1][1] - line[0][1])  # Vector 1
    for pt in pts:
        pt = np.array(pt)
        v2 = (pt[0] - line[0][0], pt[1] - line[0][1])  # Vector 1
        xp = v1[0] * v2[1] - v1[1] * v2[0]  # Cross product
        if xp > 0:
            pass
        elif xp < 0:
            if side in ("r", "b", "l", "t"):
                # below
                new_points.append(pt)
        else:
            # same
            new_points.append(pt)
    return np.array(new_points)


def split_pts(pts, border_pt, side, img_plot, debug_plot):
    if side == "t":
        border_pt_c = (border_pt[1][0] - border_pt[0][0]) / 2
    elif side == "r":
        border_pt_c = (border_pt[1][1] - border_pt[0][1]) / 2
    elif side == "b":
        border_pt_c = (border_pt[0][0] - border_pt[1][0]) / 2
    else:
        border_pt_c = (border_pt[0][1] - border_pt[1][1]) / 2

    left = []
    right = []

    for pt in pts:
        if side in ("t", "b"):
            pt_c = pt[0]
        else:
            pt_c = pt[1]

        if pt_c < border_pt_c:
            left.append(pt)
        else:
            right.append(pt)

    if debug_plot:
        for p in left:
            img_plot = cv2.circle(
                img_plot,
                (int(p[0]), int(p[1])),
                radius=10,
                color=(100, 100, 100),
                thickness=-1,
            )
        for p in right:
            img_plot = cv2.circle(
                img_plot,
                (int(p[0]), int(p[1])),
                radius=10,
                color=(255, 50, 150),
                thickness=-1,
            )
    return left, right, img_plot


class HomographyTrans:
    def __init__(self):
        self.segm = Segmentator()
        # self.tol_perc = 2.2058823529411766 * 0.01
        self.tol_perc = 5.0 * 0.01
        self.tol_orig_perc = 2.7 * 0.01

    def inference(self, img: np.array, debug_plot=True):
        main_hypot = np.hypot(img.shape[0], img.shape[1])
        tol = main_hypot * self.tol_perc
        points = self.segm.inference(img)
        if points is None:
            return img, img
        points = group_points(points, tol=tol)
        points = np.array(points)
        if debug_plot:
            h = plot_border_corrected_(img.copy(), points)

        for n, p in enumerate(points):
            h = cv2.putText(
                h,
                str(n),
                (p[0] + 50, p[1] + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # rect_all = []
        # for pi1, pi2, pi3, pi4 in combinations(range(0, points.shape[0]), r=4):
        #     p1, p2, p3, p4 = points[pi1], points[pi2], points[pi3], points[pi4]
        #     p = Polygon([p1, p2, p3, p4])
        #     rect_all.append((p.area, []))
        l_get_max_poly4 = lambda idxs: get_max_poly4(idxs, points=points)
        idxs_poly_max = max(
            combinations(range(0, points.shape[0]), r=4), key=l_get_max_poly4
        )

        if len(points) > 3:
            close_90_cluster_pts = points[[idxs_poly_max]]
            picks_clockwised = order_points_clockwise(close_90_cluster_pts)
            close_90_cluster_idxs_sorted = []
            for i, pick in enumerate(picks_clockwised):
                c_i = np.argwhere(np.sum(close_90_cluster_pts == pick, axis=1) == 2)[0][
                    0
                ]
                c = idxs_poly_max[c_i]
                close_90_cluster_idxs_sorted.append(c)

                if debug_plot:
                    h = cv2.circle(
                        h,
                        (int(pick[0]), int(pick[1])),
                        radius=10,
                        color=(255, 255, 0),
                        thickness=-1,
                    )
                    h = cv2.putText(
                        h,
                        str(int(i)),
                        (int(pick[0]) + 50, int(pick[1]) + 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
            # if debug_plot:
            #     h = cv2.polylines(
            #         h,
            #         [picks_clockwised.astype(int)],
            #         True,
            #         color=(0, 255, 255),
            #         thickness=5,
            #     )
            if debug_plot:
                h = cv2.polylines(
                    h,
                    [picks_clockwised.astype(int)],
                    True,
                    color=(0, 255, 0),
                    thickness=5,
                )
                # Image.fromarray(h[:, :, ::-1]).show()
            # add square around the polygon

            # choose direction
            l = len(points)
            idxs_top = polyg(
                close_90_cluster_idxs_sorted[0], close_90_cluster_idxs_sorted[1], l=l
            )
            idxs_right = polyg(
                close_90_cluster_idxs_sorted[1], close_90_cluster_idxs_sorted[2], l=l
            )
            idxs_bot = polyg(
                close_90_cluster_idxs_sorted[2], close_90_cluster_idxs_sorted[3], l=l
            )
            idxs_left = polyg(
                close_90_cluster_idxs_sorted[3], close_90_cluster_idxs_sorted[0], l=l
            )

            top = below_above_line_filter(
                (picks_clockwised[0], picks_clockwised[1]), points[idxs_top], side="t"
            )
            right = below_above_line_filter(
                (picks_clockwised[1], picks_clockwised[2]), points[idxs_right], side="r"
            )

            bot = below_above_line_filter(
                (picks_clockwised[2], picks_clockwised[3]), points[idxs_bot], side="b"
            )
            left = below_above_line_filter(
                (picks_clockwised[3], picks_clockwised[0]), points[idxs_left], side="l"
            )

            top_l, top_r, h = split_pts(top,
                                     (picks_clockwised[0], picks_clockwised[1]),
                                     side="t", img_plot=h, debug_plot=debug_plot)
            right_l, right_r, h = split_pts(right,
                                         (picks_clockwised[1], picks_clockwised[2]),
                                         side="r", img_plot=h, debug_plot=debug_plot)
            bot_l, bot_r, h = split_pts(bot,
                                     (picks_clockwised[2], picks_clockwised[3]),
                                     side="b", img_plot=h, debug_plot=debug_plot)
            left_l, left_r, h = split_pts(left,
                                       (picks_clockwised[3], picks_clockwised[0]),
                                       side="l", img_plot=h, debug_plot=debug_plot)
            if debug_plot:
                # Image.fromarray(h[:, :, ::-1]).show()
                pass

            picks_clockwised[0][0] = np.min([picks_clockwised[0][0], ] + [_[0] for _ in top_l] + [_[0] for _ in left_l])
            picks_clockwised[0][1] = np.min([picks_clockwised[0][1], ] + [_[1] for _ in top_l] + [_[1] for _ in left_l])
            picks_clockwised[1][0] = np.max([picks_clockwised[1][0], ] + [_[0] for _ in top_r] + [_[0] for _ in right_l])
            picks_clockwised[1][1] = np.min([picks_clockwised[1][1], ] + [_[1] for _ in top_r] + [_[1] for _ in right_l])
            picks_clockwised[2][0] = np.max([picks_clockwised[2][0], ] + [_[0] for _ in right_r] + [_[0] for _ in bot_r])
            picks_clockwised[2][1] = np.max([picks_clockwised[2][1], ] + [_[1] for _ in right_r] + [_[1] for _ in bot_r])
            picks_clockwised[3][0] = np.min([picks_clockwised[3][0], ] + [_[0] for _ in bot_l] + [_[0] for _ in left_r])
            picks_clockwised[3][1] = np.max([picks_clockwised[3][1], ] + [_[1] for _ in bot_l] + [_[1] for _ in left_r])

            if debug_plot:
                h = cv2.polylines(
                    h,
                    [picks_clockwised.astype(int)],
                    True,
                    color=(0, 0, 255),
                    thickness=5,
                )
                # Image.fromarray(h[:, :, ::-1]).show()

            dist_pt1 = (
                abs(picks_clockwised[0][0] - 0),
                abs(picks_clockwised[0][1] - 0),
            )
            dist_pt2 = (
                abs(picks_clockwised[1][0] - img.shape[1]),
                abs(picks_clockwised[1][1] - 0),
            )
            dist_pt3 = (
                abs(picks_clockwised[2][0] - img.shape[1]),
                abs(picks_clockwised[2][1] - img.shape[0]),
            )
            dist_pt4 = (
                abs(picks_clockwised[3][0] - 0),
                abs(picks_clockwised[3][1] - img.shape[0]),
            )
            tol_orig = self.tol_orig_perc * main_hypot
            if dist_pt1[0] < tol_orig:
                picks_clockwised[0][0] = 0
            if dist_pt1[1] < tol_orig:
                picks_clockwised[0][1] = 0
            if dist_pt2[0] < tol_orig:
                picks_clockwised[1][0] = img.shape[1]
            if dist_pt2[1] < tol_orig:
                picks_clockwised[1][1] = 0
            if dist_pt3[0] < tol_orig:
                picks_clockwised[2][0] = img.shape[1]
            if dist_pt3[1] < tol_orig:
                picks_clockwised[2][1] = img.shape[0]
            if dist_pt4[0] < tol_orig:
                picks_clockwised[3][0] = 0
            if dist_pt4[1] < tol_orig:
                picks_clockwised[3][1] = img.shape[0]

            if debug_plot:
                h = cv2.polylines(
                    h,
                    [picks_clockwised.astype(int)],
                    True,
                    color=(21, 244, 252),
                    thickness=5,
                )
                # Image.fromarray(h[:, :, ::-1]).show()

            un_warped, cropped = dewarp(
                image=img, pts_src=picks_clockwised, debug_plot=debug_plot
            )
            if debug_plot:
                # Image.fromarray(cropped[:, :, ::-1]).show()
                pass
            return cropped, h
        return img, img


if __name__ == "__main__":
    import json
    import os
    import os.path as osp
    from glob import glob
    from tqdm import tqdm

    ht = HomographyTrans()
    base = os.getcwd()
    # imgs = glob("/Users/dmitry/Initflow/doc-img-dewarping/output_orig/*")
    # imgs = glob("/Users/dmitry/Initflow/doc-img-dewarping/imgs_bugs/*")
    imgs = glob("/Users/dmitry/Initflow/doc-img-dewarping/content/imgs_test_ht/*")
    # imgs = glob(osp.join(base, "test_imgs/*"))
    # anns_dir = osp.join(base, "test_anns_4_pts/")
    anns_dir = osp.join(base, "test_anns/")
    for im_path in tqdm(imgs):
        # if (
        #     # "rot_invoices(apr1-10)-001_21042071_6SbTDzIQ2MnCP7Vj1ZJq_0.jpg"
        #     "rot_invoices(apr1-10)-001_21006951_1Ooa07GmaxGJsTWesOd2_0.jpg"
        #     not in im_path
        # ):
        #     continue
        im_fn = osp.basename(im_path)
        img = cv2.imread(im_path)

        out_img, h = ht.inference(img=img, debug_plot=True)
        cv2.imwrite(
            osp.join(
                "/Users/dmitry/Initflow/doc-img-dewarping/content/segm_out",
                osp.basename(im_path),
            ),
            out_img,
        )
        cv2.imwrite(
            osp.join(
                "/Users/dmitry/Initflow/doc-img-dewarping/content/segm_out_red_poly",
                osp.basename(im_path),
            ),
            h,
        )
