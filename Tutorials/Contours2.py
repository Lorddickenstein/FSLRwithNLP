import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils

def show_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_plt_image(src_img):
    plt.imshow(src_img)
    plt.show()

def resize_image(src_img, img_size):
    return cv2.resize(src_img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

def get_thresh(src_img):
    return cv2.threshold(src_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

def morph_image(src_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(src_img, cv2.MORPH_CLOSE, kernel)

def get_contours(img):
    return cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def get_bounding_rect(src_img, mask):
    pts = np.column_stack(np.where(mask.transpose() > 0))
    x, y, w, h = cv2.boundingRect(pts)
    return cv2.rectangle(src_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

def get_bounding_rotated_rect(src_img, mask):
    pts = np.column_stack(np.where(mask.transpose() > 0))
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(src_img, [box], 0, (0, 0, 255), 2)

def get_convex_hull(hull, src_img):
    pts = np.column_stack(np.where(src_img.transpose() > 0))
    hullpts = cv2.convexHull(pts)
    ((centx, centy), (width, height), angle) = cv2.fitEllipse(hullpts)

    # Draw convex hull on image
    cv2.polylines(hull, [hullpts], True, (0, 0, 255), 1)
    return hull


# img = cv2.imread('D:\Documents\Thesis\OurDataset\Raw Dataset\A\color_0_1.jpg')
img = cv2.imread('D:\Documents\Thesis\FSLRwithNLP\Datasets\Test_Images\L2.jpg')
show_image('original', img)
imgCopy = img.copy()
blank = np.zeros(imgCopy.shape, dtype='uint8')

skin_mask = utils.segmentation(imgCopy)
gray_img = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2GRAY)

blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
# blur_img = cv2.bilateralFilter(imgCopy, 5, 10, 10)

_, th = get_thresh(blur_img)
show_plt_image(th)

morph = morph_image(th)
edges = utils.get_edges(morph)


# Inverse morph
mask = 255 - morph
# show_image('mask', mask)

# Apply mask
result = cv2.bitwise_and(imgCopy, imgCopy, mask=mask)
# show_plt_image(result)

# edges = cv2.Canny(morph, 150, 210)
contours, hierarchies = get_contours(edges)
cnt = max(contours, key=lambda x: cv2.contourArea(x))
cv2.drawContours(blank, contours, -1, (255,255,255), 2)
show_plt_image(blank)

# Get rectangle
rectangle = get_bounding_rect(imgCopy, mask)
# rectangle = get_bounding_rotated_rect(imgCopy, mask)

# show_plt_image(imgCopy)
show_plt_image(imgCopy)

hull = get_convex_hull(img.copy(), mask)
show_plt_image(hull)

