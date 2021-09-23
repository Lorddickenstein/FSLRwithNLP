import numpy as np
import cv2
import os

# Show image
def show_image(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

path = "D:\Documents\Thesis\FSLRwithNLP\Datasets\Test_Images"
file_name = "Y2.jpg"
# img = cv2.imread(os.path.join(path, file_name), 0)
img = cv2.imread(os.path.join(path, file_name), 1)

# Blur image using Gaussian Blur
blur_img = cv2.GaussianBlur(img, (5, 5), 0)

# Normalize image
norm_img = blur_img.astype(np.float32) / 255.0

# Edge Detection
"""Pretrained Forest Model"""
edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("models/model.yml")
edges = edgeDetector.detectEdges(norm_img) * 255.0
"""Regular Canny Edge Detection with Otsu's Binarization"""
# _, th = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# edges = cv2.Canny(th, 150, 210)

# show_image(edges, 'edges')

def SaltPepperNoise(edges):
    count = 0
    lastMedian = edges
    median = cv2.medianBlur(edges, 3)
    while not np.array_equal(lastMedian, median):
        zeroed = np.invert(np.logical_and(median, edges))
        edges[zeroed] = 0
        count = count + 1
        if count > 70:
            break
        lastMedian = median
        median = cv2.medianBlur(edges, 3)

edges_ = np.asarray(edges, np.uint8)
SaltPepperNoise(edges_)
# show_image(edges_, "IDK")

def findSignificantContour(edgeImg):
    contours, hierarchy = cv2.findContours(
        edgeImg,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    level1Meta = []
    for contourIndex, tupl in enumerate(hierarchy[0]):
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)
# From among them, find the contours with large surface area.
    contoursWithArea = []
    for tupl in level1Meta:
        contourIndex = tupl[0]
        contour = contours[contourIndex]
        area = cv2.contourArea(contour)
        contoursWithArea.append([contour, area, contourIndex])

    contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour

contour = findSignificantContour(edges_)
# Draw the contour on the original image
contourImg = np.copy(contour)
cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
# show_image(contourImg, "contour")

mask = np.zeros_like(contourImg)
cv2.fillPoly(mask, [contour], 255)
# calculate sure foreground area by dilating the mask
mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)
# mark inital mask as "probably background"
# and mapFg as sure foreground
trimap = np.copy(mask)
trimap[mask == 0] = cv2.GC_BGD
trimap[mask == 255] = cv2.GC_PR_BGD
trimap[mapFg == 255] = cv2.GC_FGD
# visualize trimap
trimap_print = np.copy(trimap)
trimap_print[trimap_print == cv2.GC_PR_BGD] = 128
trimap_print[trimap_print == cv2.GC_FGD] = 255
cv2.imwrite('trimap.png', trimap_print)