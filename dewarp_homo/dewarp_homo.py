import cv2
import numpy as np


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
        pts = np.array(points).astype(int)

        un_warped = dewarp(image=img, pts_src=pts)
        Image.fromarray(un_warped[:, :, ::-1]).show()
