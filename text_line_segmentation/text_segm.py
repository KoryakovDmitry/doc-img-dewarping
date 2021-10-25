import cv2
from PIL import Image
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

img = cv2.imread("../test_imgs/33.jpg")
mser = cv2.MSER_create()

# Resize the image so that MSER can work better
img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_denoised = cv2.fastNlMeansDenoising(gray)
vis = img_denoised.copy()

regions = mser.detectRegions(img_denoised)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]

cv2.polylines(vis, hulls, 1, (0, 255, 0))

Image.fromarray(vis).show()
