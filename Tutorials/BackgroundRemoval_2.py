import cv2
import numpy as np

# Read image
img = cv2.imread('D:\Documents\Thesis\FSLRwithNLP\Datasets\Test_Images\spell.27.jpg')
hh, ww = img.shape[:2]
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold on white
# Define lower and upper limits
# lower = np.array([200, 200, 200])
# upper = np.array([255, 255, 255])

# blur
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

# Create mask to only select black
# thresh = cv2.inRange(img, lower, upper)
_, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# apply morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# invert morph image
# mask = 255 - morph

# apply mask to image
result = cv2.bitwise_and(img, img, mask=morph)

# cv2.imwrite('Y2_thresh.jpg', thresh)
# cv2.imwrite('Y2_morph.jpg', morph)
# cv2.imwrite('Y2_mask.jpg', mask)
# cv2.imwrite('Y2_result.jpg', result)

cv2.imshow('thresh', thresh)
cv2.imshow('morph', morph)
cv2.imshow('mask', morph)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()