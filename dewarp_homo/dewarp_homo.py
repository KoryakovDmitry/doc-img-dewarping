import cv2
import math
from copy import deepcopy
import numpy as np
from sympy.geometry import Line, Point2D
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed


def order_points_clockwise(pts):
    rect = np.zeros((4, 2))
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def k_means_res(scaled_data, k, alpha_k=0.02):
    """
    Parameters
    ----------
    scaled_data: matrix
        scaled data. rows are samples and columns are features for clustering
    k: int
        current k for applying KMeans
    alpha_k: float
        manually tuned factor that gives penalty to the number of clusters
    Returns
    -------
    scaled_inertia: float
        scaled inertia value for current k
    """

    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    # fit k-means
    kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return scaled_inertia, kmeans.labels_


def choose_best_k_for_k_means_parallel(scaled_data, k_range):
    """
    Parameters
    ----------
    scaled_data: matrix
        scaled data. rows are samples and columns are features for clustering
    k_range: list of integers
        k range for applying KMeans
    Returns
    -------
    best_k: int
        chosen value of k out of the given k range.
        chosen k is k with the minimum scaled inertia value.
    results: pandas DataFrame
        adjusted inertia value for each k in k_range
    """

    ans = Parallel(n_jobs=-1, verbose=10, backend="multiprocessing")(
        delayed(k_means_res)(scaled_data, k) for k in k_range
    )
    ans = list(zip(k_range, (_[0] for _ in ans), (_[1] for _ in ans)))
    best_k = min(ans, key=lambda k: k[1])
    return best_k, ans


def to_cluster(data_, range_cluster=10):
    data = np.array([_[0] for _ in data_])[:, None]

    # scale the data
    mms = MinMaxScaler()
    scaled_data = mms.fit_transform(data)

    # choose k range
    k_range = range(2, range_cluster)
    # compute adjusted intertia
    best_k, results = choose_best_k_for_k_means_parallel(scaled_data, k_range)
    lbls = best_k[-1]

    # # DEBUG plot the results
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(7, 4))
    # plt.scatter(x=[_[0] for _ in results], y=[_[1] for _ in results])
    # plt.title('Adjusted Inertia for each K')
    # plt.xlabel('K')
    # plt.ylabel('Adjusted Inertia')
    # plt.xticks(range(2, range_cluster, 1))
    # plt.show()

    clusters = {l: [] for l in lbls if l != -1}
    for d, l in zip(data_, lbls):
        if l != -1:
            clusters[l].append(d)
    return clusters


def sort_clockwise(points_inp):
    points = [Point2D(p) for p in points_inp]
    if len(points) > 1:
        points_sorted = []
        pt_start_i = 0

        already = {
            0,
        }
        while True:
            d = []
            for pt_i in range(len(points)):
                if (pt_start_i != pt_i) and (pt_i not in already):
                    d.append((pt_i, points[pt_start_i].distance(points[pt_i])))

            p_i = min(d, key=lambda x: x[-1])[0]
            points_sorted.append(points_inp[p_i])
            already.add(p_i)
            pt_start_i = p_i
            if len(already) == len(points_inp):
                break

        return points_sorted
    else:
        return points_inp


def get_destination_points(corners):
    """
    -Get destination points from corners of warped images
    -Approximating height and width of the rectangle: we take maximum of the 2 widths and 2 heights

    Args:
        corners: list

    Returns:
        destination_corners: list
        height: int
        width: int

    """

    w1 = np.hypot(
        abs(corners[0][0] - corners[1][0]), abs(corners[0][1] - corners[1][1])
    )
    w2 = np.hypot(
        abs(corners[2][0] - corners[3][0]), abs(corners[2][1] - corners[3][1])
    )
    w = max(int(w1), int(w2))

    h1 = np.hypot(
        abs(corners[0][0] - corners[2][0]), abs(corners[0][1] - corners[2][1])
    )
    h2 = np.hypot(
        abs(corners[1][0] - corners[3][0]), abs(corners[1][1] - corners[3][1])
    )
    h = max(int(h1), int(h2))

    destination_corners = np.float32([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])
    return destination_corners, h, w


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
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    un_warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)
    return un_warped


def dewarp(image, pts_src):
    destination_points, h, w = get_destination_points(pts_src)
    un_warped = unwarp(image, np.float32(pts_src), destination_points)
    return un_warped


def get_angle_between_two_lines(first_line, second_line):
    first_line = Line(Point2D(first_line[0]), Point2D(first_line[1]))
    second_line = Line(Point2D(second_line[0]), Point2D(second_line[1]))
    return 180 - np.rad2deg(float(first_line.angle_between(second_line)))


if __name__ == "__main__":
    import cv2
    from PIL import Image
    import numpy as np
    import json
    import os
    import os.path as osp
    from glob import glob

    base = os.getcwd()
    imgs = glob(osp.join(base, "test_imgs/*"))
    anns_dir = osp.join(base, "test_anns/")
    for im_path in imgs:
        im_fn = osp.basename(im_path)
        img = cv2.imread(im_path)

        with open(osp.join(anns_dir, im_fn.replace(".jpg", ".json"))) as f:
            anns = json.load(f)

        points = anns["shapes"][0]["points"]
        # points = np.random.permutation(np.array(points)).tolist()

        img_plot = img.copy()
        for n, p in enumerate(points):
            p = np.array(p).astype(int)
            img_plot = cv2.circle(
                img_plot, (p[0], p[1]), radius=10, color=(255, 255, 0), thickness=-1
            )
            img_plot = cv2.putText(
                img_plot,
                str(n + 1),
                (p[0] + 50, p[1] + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # Image.fromarray(img_plot[:, :, ::-1]).show()

        # points = sort_clockwise(points)
        # img_plot = img.copy()
        # for n, p in enumerate(points):
        #     p = np.array(p).astype(int)
        #     img_plot = cv2.circle(
        #         img_plot, (p[0], p[1]), radius=10, color=(255, 255, 0), thickness=-1
        #     )
        #     img_plot = cv2.putText(
        #         img_plot,
        #         str(n + 1),
        #         (p[0] + 50, p[1] + 50),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         (255, 0, 0),
        #         2,
        #         cv2.LINE_AA,
        #     )
        #
        # Image.fromarray(img_plot[:, :, ::-1]).show()

        if len(points) > 2:
            angles = []
            for prev_i in range(0, len(points) - 2):
                prev_pt = points[prev_i]
                now_pt = points[prev_i + 1]
                next_pt = points[prev_i + 2]

                first_line = [prev_pt, now_pt]
                second_line = [now_pt, next_pt]

                l_1 = np.array(first_line).astype(int)
                l_2 = np.array(second_line).astype(int)
                img_plot = cv2.line(
                    img_plot, tuple(l_1[0]), tuple(l_1[1]), (255, 0, 255), thickness=3
                )
                img_plot = cv2.line(
                    img_plot, tuple(l_2[0]), tuple(l_2[1]), (255, 0, 255), thickness=3
                )

                angle = get_angle_between_two_lines(first_line, second_line)
                angles.append(
                    (
                        abs(90 - angle),
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
            img_plot = cv2.line(
                img_plot, tuple(l_1[0]), tuple(l_1[1]), (255, 255, 0), thickness=3
            )
            img_plot = cv2.line(
                img_plot, tuple(l_2[0]), tuple(l_2[1]), (255, 255, 0), thickness=3
            )

            angle = get_angle_between_two_lines(first_line, second_line)
            angles.append(
                (
                    abs(90 - angle),
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
            img_plot = cv2.line(
                img_plot, tuple(l_1[0]), tuple(l_1[1]), (0, 255, 255), thickness=3
            )
            img_plot = cv2.line(
                img_plot, tuple(l_2[0]), tuple(l_2[1]), (0, 255, 255), thickness=3
            )

            angle = get_angle_between_two_lines(first_line, second_line)
            angles.append(
                (abs(90 - angle), (prev_i + 2, 0, 1), (first_line, second_line))
            )

            clusters = to_cluster(
                angles, range_cluster=len(angles) if len(angles) > 10 else len(angles)
            )

            close_90_cluster = clusters[
                min(
                    clusters,
                    key=lambda l: np.mean([_[0] for _ in clusters[l]]),
                )
            ]

            for i in close_90_cluster:
                l_1 = np.array(i[-1][0]).astype(int)
                l_2 = np.array(i[-1][1]).astype(int)
                img_plot = cv2.line(
                    img_plot, tuple(l_1[0]), tuple(l_1[1]), (0, 255, 0), thickness=3
                )
                img_plot = cv2.line(
                    img_plot, tuple(l_2[0]), tuple(l_2[1]), (0, 255, 0), thickness=3
                )
                img_plot = cv2.putText(
                    img_plot,
                    str(int(i[0])),
                    (l_1[1][0] + 50, l_1[1][1] + 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Image.fromarray(img_plot[:, :, ::-1]).show()
            assert len(close_90_cluster) == 4, f"need 4 pts not {len(close_90_cluster)}"

            close_90_cluster_pts = np.array([_[-1][0][1] for _ in close_90_cluster])
            picks_clockwised = order_points_clockwise(close_90_cluster_pts)
            close_90_cluster_sorted = []
            for pick in picks_clockwised:
                c_i = np.argwhere(np.sum(close_90_cluster_pts == pick, axis=1) == 2)[0][0]
                c = close_90_cluster[c_i]
                close_90_cluster_sorted.append(c)

            for p in picks_clockwised:
                p = np.array(p).astype(int)
                img_plot = cv2.circle(
                    img_plot, (p[0], p[1]), radius=10, color=(0, 0, 255), thickness=-1
                )

            Image.fromarray(img_plot[:, :, ::-1]).show()

            h = None
        # pts = np.array(points).astype(int)
        #
        # for p in range(len(pts)):

        # un_warped = dewarp(image=img, pts_src=pts)
        # Image.fromarray(un_warped[:, :, ::-1]).show()
