import cv2
import math
from copy import deepcopy
import numpy as np
from sympy.geometry import Line, Point2D
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed


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
                color=colors[side],
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

    save_to = f"/Users/dmitry/Initflow/doc-img-dewarping/dewarp_homo/"
    name = str(points_[0]).replace(",", "").replace(".", "").replace(" ", "").replace("[", "").replace("]", "")+".jpg"
    cv2.imwrite(osp.join(save_to, name), img_plot)
    return img_plot


colors = {
    "t": (125, 125, 0),
    "r": (100, 200, 150),
    "b": (255, 0, 255),
    "l": (125, 125, 125),
}


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


def get_destination_points(points, border, img_plot):
    assert (
        points[0] == border[0][-1][0][1]
    ), f"need the same start point {points[0]} {border[0][-1][0][1]}"

    destination_pts = {"t": [], "r": [], "b": [], "l": []}
    clock_idxs = [c[1][1] for c in border if c[1][1] != 0]
    points = np.array(points)
    reverse_direction_x_y = 0
    destination_pts["t"].append([0, 0])
    prev = [0, 0]
    shapes = {"t": 0, "r": 0, "l": 0, "b": 0}

    for pt_i_next in range(1, len(points)):
        pt_i = pt_i_next - 1
        pt = points[pt_i]
        pt_next = points[pt_i_next]

        if pt_i in clock_idxs:
            reverse_direction_x_y += 1

        hypot = np.hypot(
            abs(pt[0] - pt_next[0]),
            abs(pt[1] - pt_next[1]),
        )

        # img_plot_ = img_plot.copy()
        #
        # p = np.array(pt).astype(int)
        # img_plot_ = cv2.circle(
        #     img_plot_,
        #     (p[0], p[1]),
        #     radius=10,
        #     color=(200, 100, 255),
        #     thickness=-1,
        # )
        #
        # p = np.array(pt_next).astype(int)
        # img_plot_ = cv2.circle(
        #     img_plot_,
        #     (p[0], p[1]),
        #     radius=10,
        #     color=(200, 100, 255),
        #     thickness=-1,
        # )
        # Image.fromarray(img_plot_[:, :, ::-1]).show()

        if reverse_direction_x_y == 0:
            shapes["t"] += hypot
            destination_pts["t"].append([prev[0] + hypot, 0])
            prev = [prev[0] + hypot, 0]
        elif reverse_direction_x_y == 1:
            shapes["r"] += hypot
            destination_pts["r"].append([prev[0] - 1, prev[1] + hypot])
            prev = [prev[0] - 1, prev[1] + hypot]
        elif reverse_direction_x_y == 2:
            shapes["b"] += hypot
            destination_pts["b"].append([prev[0] - hypot, prev[1] - 1])
            prev = [prev[0] - hypot, prev[1] - 1]
        elif reverse_direction_x_y == 3:
            shapes["l"] += hypot
            destination_pts["l"].append([prev[0] - 1, prev[1] - hypot])
            prev = [prev[0] - 1, prev[1] - hypot]
        else:
            h = None
    else:
        pt_i = 0
        pt = points[pt_i]
        pt_next = points[pt_i_next]

        hypot = np.hypot(
            abs(pt[0] - pt_next[0]),
            abs(pt[1] - pt_next[1]),
        )
        shapes["l"] += hypot
        destination_pts["l"].append([prev[0] - 1, prev[1] - hypot])

    # img_plot = cv2.copyMakeBorder(
    #     img_plot, 500, 500, 500, 500, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    # )

    # img_plot_ = plot_border_corrected(
    #     img_plot, destination_pts, points, add_x=1000, add_y=500
    # )

    delta_w = abs(shapes["t"] - shapes["b"])
    delta_h = abs(shapes["l"] - shapes["r"])
    max_w_side_name = max(shapes, key=lambda x: shapes[x] if x in ("t", "b") else -1)
    max_h_side_name = max(shapes, key=lambda x: shapes[x] if x in ("l", "r") else -1)

    min_w_side_name = "b" if max_w_side_name == "t" else "t"
    min_h_side_name = "r" if max_h_side_name == "l" else "l"

    l_d_w = len(destination_pts["t"]) - 1
    l_d_h = len(destination_pts["l"])
    # l_d_w = len(destination_pts[min_w_side_name])
    # l_d_h = len(destination_pts[min_h_side_name])

    # l_d_w = l_d_w - 1 if min_w_side_name == "t" else l_d_w
    # l_d_h = l_d_h if min_h_side_name == "r" else l_d_h

    step_w = delta_w / l_d_w
    step_h = delta_h / l_d_h

    if min_w_side_name == "b":
        for i in range(l_d_w):
            destination_pts["t"][i][0] += step_w * (i + 1)

    elif min_w_side_name == "t":
        for n, i in enumerate(reversed(range(l_d_w))):
            destination_pts[min_w_side_name][i][0] -= step_w * (n + 1)

    if min_h_side_name == "r":
        for n, i in enumerate(reversed(range(l_d_h))):
            destination_pts["l"][i][1] += step_h * (n + 1)

    elif min_h_side_name == "l":
        for i in range(l_d_h):
            destination_pts[min_h_side_name][i][1] -= step_h * (i + 1)
    #

    destination_pts["t"][0][0] = destination_pts["l"][-1][0]
    destination_pts["l"] = destination_pts["l"][:-1]

    # img_plot = plot_border_corrected(
    #     img_plot, destination_pts, points, add_x=1000, add_y=500
    # )

    # img_plot = plot_border_corrected(
    #     img_plot, destination_pts, points, add_x=500, add_y=0
    # )
    destination_pts_list = [i for l in destination_pts for i in destination_pts[l]]
    destination_pts_list = np.array(destination_pts_list)
    destination_pts_list_neg_min = np.min(destination_pts_list, axis=0)
    destination_pts_list_neg_min = np.array(
        [abs(_) if _ < 0 else 0 for _ in destination_pts_list_neg_min]
    )
    destination_pts_list = destination_pts_list + destination_pts_list_neg_min + 1
    img_plot = plot_border_corrected(
        img_plot, {"l": destination_pts_list}, points, add_x=0, add_y=0
    )

    return destination_pts_list, shapes[max_h_side_name], shapes[max_w_side_name]


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
    destination_pts, h, w = get_destination_points(
        points=points, border=close_90_cluster_sorted, img_plot=img_plot
    )
    h, w = int(np.round(h)), int(np.round(w))
    un_warped = unwarp(image, np.float32(pts_src), np.float32(destination_pts))
    cropped = un_warped[0:h, 0:w]
    return un_warped, cropped


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
                str(n),
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
                c_i = np.argwhere(np.sum(close_90_cluster_pts == pick, axis=1) == 2)[0][
                    0
                ]
                c = close_90_cluster[c_i]
                close_90_cluster_sorted.append(c)

            for p in picks_clockwised:
                p = np.array(p).astype(int)
                img_plot = cv2.circle(
                    img_plot, (p[0], p[1]), radius=10, color=(0, 0, 255), thickness=-1
                )

            # Image.fromarray(img_plot[:, :, ::-1]).show()

            # dewarp(image=img, pts_src=points)
            un_warped, cropped = dewarp(image=img, pts_src=points)

            Image.fromarray(un_warped[:, :, ::-1]).show()
            Image.fromarray(cropped[:, :, ::-1]).show()
