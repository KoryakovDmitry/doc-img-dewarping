from PIL import Image
import imutils
import cv2
# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread("/Users/dmitry/Initflow/doc-img-dewarping/test.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# threshold the image,
thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
# find contours in thresholded image, then grab the largest
# one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)
# draw the contours of c
cv2.drawContours(image, [c], -1, (0, 0, 255), 2)

# show the output image
Image.fromarray(image[:, :, ::-1]).show()

h = None