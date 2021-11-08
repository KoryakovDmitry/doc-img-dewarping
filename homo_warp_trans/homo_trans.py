import math
from copy import deepcopy
import numpy as np
from sympy.geometry import Line, Point2D

# from segmentation import (
from .segmentation import (
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


class HomographyTrans:
    def __init__(self):
        self.segm = Segmentator()
        # self.tol_perc = 2.2058823529411766 * 0.01
        self.tol_perc = 5.0 * 0.01
        self.tol_orig_perc = 2.5 * 0.01

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
            angles = []

            if debug_plot:
                img_plot = img.copy()

            for prev_i in range(0, len(points) - 2):
                prev_pt = points[prev_i]
                now_pt = points[prev_i + 1]
                next_pt = points[prev_i + 2]

                first_line = [prev_pt, now_pt]
                second_line = [now_pt, next_pt]

                l_1 = np.array(first_line).astype(int)
                l_2 = np.array(second_line).astype(int)

                if debug_plot:
                    img_plot = cv2.line(
                        img_plot,
                        tuple(l_1[0]),
                        tuple(l_1[1]),
                        (255, 0, 255),
                        thickness=3,
                    )
                    img_plot = cv2.line(
                        img_plot,
                        tuple(l_2[0]),
                        tuple(l_2[1]),
                        (255, 0, 255),
                        thickness=3,
                    )

                angles.append(
                    (
                        (prev_i, prev_i + 1, prev_i + 2),
                        (first_line, second_line),
                    )
                )

            prev_pt = points[prev_i + 1]
            now_pt = points[prev_i + 2]
            next_pt = points[0]

            first_line = [prev_pt, now_pt]
            second_line = [now_pt, next_pt]

            l_1 = np.array(first_line).astype(int)
            l_2 = np.array(second_line).astype(int)
            if debug_plot:
                img_plot = cv2.line(
                    img_plot, tuple(l_1[0]), tuple(l_1[1]), (255, 255, 0), thickness=3
                )
                img_plot = cv2.line(
                    img_plot, tuple(l_2[0]), tuple(l_2[1]), (255, 255, 0), thickness=3
                )

            angles.append(
                (
                    (prev_i + 1, prev_i + 2, 0),
                    (first_line, second_line),
                )
            )

            prev_pt = points[prev_i + 2]
            now_pt = points[0]
            next_pt = points[1]

            first_line = [prev_pt, now_pt]
            second_line = [now_pt, next_pt]

            l_1 = np.array(first_line).astype(int)
            l_2 = np.array(second_line).astype(int)

            if debug_plot:
                img_plot = cv2.line(
                    img_plot, tuple(l_1[0]), tuple(l_1[1]), (0, 255, 255), thickness=3
                )
                img_plot = cv2.line(
                    img_plot, tuple(l_2[0]), tuple(l_2[1]), (0, 255, 255), thickness=3
                )

            angles.append(((prev_i + 2, 0, 1), (first_line, second_line)))

            poly_max_ang = [get_angle_by_idx(angles, idx) for idx in idxs_poly_max]

            close_90_cluster_pts = points[[idxs_poly_max]]
            picks_clockwised = order_points_clockwise(close_90_cluster_pts)
            close_90_cluster_sorted = []
            for pick in picks_clockwised:
                c_i = np.argwhere(np.sum(close_90_cluster_pts == pick, axis=1) == 2)[0][
                    0
                ]
                c = poly_max_ang[c_i]
                close_90_cluster_sorted.append(c)

            # change near pts (~ 90 deg)
            for num, c in enumerate(close_90_cluster_sorted):
                pts = c[1][0][1]
                pts_l = c[1][0][0]
                pts_r = c[1][1][1]

                if num == 0:
                    picks_clockwised[num][0] = np.min([pts_l[0], pts[0], pts_r[0]])
                    picks_clockwised[num][1] = np.min([pts_l[1], pts[1], pts_r[1]])

                elif num == 1:
                    picks_clockwised[num][0] = np.max([pts_l[0], pts[0], pts_r[0]])
                    picks_clockwised[num][1] = np.min([pts_l[1], pts[1], pts_r[1]])

                elif num == 2:
                    picks_clockwised[num][0] = np.max([pts_l[0], pts[0], pts_r[0]])
                    picks_clockwised[num][1] = np.max([pts_l[1], pts[1], pts_r[1]])

                elif num == 3:
                    picks_clockwised[num][0] = np.min([pts_l[0], pts[0], pts_r[0]])
                    picks_clockwised[num][1] = np.max([pts_l[1], pts[1], pts_r[1]])

            if debug_plot:
                h = cv2.polylines(
                    h,
                    [picks_clockwised.astype(int)],
                    True,
                    color=(0, 0, 255),
                    thickness=5,
                )
                # Image.fromarray(h[:, :, ::-1]).show()

            dist_pt1 = np.hypot(
                abs(picks_clockwised[0][0] - 0),
                abs(picks_clockwised[0][1] - 0),
            )
            dist_pt2 = np.hypot(
                abs(picks_clockwised[1][0] - img.shape[1]),
                abs(picks_clockwised[1][1] - 0),
            )
            dist_pt3 = np.hypot(
                abs(picks_clockwised[2][0] - img.shape[1]),
                abs(picks_clockwised[2][1] - img.shape[0]),
            )
            dist_pt4 = np.hypot(
                abs(picks_clockwised[3][0] - 0),
                abs(picks_clockwised[3][1] - img.shape[0]),
            )
            tol_orig = self.tol_orig_perc * main_hypot
            if dist_pt1 < tol_orig:
                picks_clockwised[0][0] = 0
                picks_clockwised[0][1] = 0
            if dist_pt2 < tol_orig:
                picks_clockwised[1][0] = img.shape[1]
                picks_clockwised[1][1] = 0
            if dist_pt3 < tol_orig:
                picks_clockwised[2][0] = img.shape[1]
                picks_clockwised[2][1] = img.shape[0]
            if dist_pt4 < tol_orig:
                picks_clockwised[3][0] = 0
                picks_clockwised[3][1] = img.shape[0]

            if debug_plot:
                h = cv2.polylines(
                    h,
                    [picks_clockwised.astype(int)],
                    True,
                    color=(0, 255, 0),
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

    ht = HomographyTrans()
    base = os.getcwd()
    # imgs = glob("/Users/dmitry/Initflow/doc-img-dewarping/output_orig/*")
    imgs = glob("/Users/dmitry/Initflow/doc-img-dewarping/imgs_bugs/*PaPd*")
    # imgs = glob(osp.join(base, "test_imgs/*"))
    # anns_dir = osp.join(base, "test_anns_4_pts/")
    anns_dir = osp.join(base, "test_anns/")
    for im_path in imgs:
        # if "rot_invoices(apr1-10)-001_00113086_kAePHqov8KWFX82RrdRT_0" not in im_path:
        #     continue
        im_fn = osp.basename(im_path)
        img = cv2.imread(im_path)

        out_img, h = ht.inference(img=img, debug_plot=True)
        cv2.imwrite(
            osp.join(
                "/Users/dmitry/Initflow/doc-img-dewarping/segm_out",
                osp.basename(im_path),
            ),
            out_img,
        )
        cv2.imwrite(
            osp.join(
                "/Users/dmitry/Initflow/doc-img-dewarping/segm_out_red_poly",
                osp.basename(im_path),
            ),
            h,
        )
