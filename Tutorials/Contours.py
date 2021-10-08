import numpy as np
import cv2
import matplotlib.pyplot as plt
import Application.HandTrackingModule as HTM
import Application.utils as utils

def show_image(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_plt_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def morph_image(src_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(src_img, cv2.MORPH_CLOSE, kernel)

def resize_image(img):
    img_size = 224
    return cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

# cv2.RETR_TREE - find all hierarchical contours
# cv2.RETR_EXTERNAL - get the external contours
# cv2.RETR_LIST - all contours of the image

# cv2.CHAIN_APPROX_NONE - all boundary points are stored
# cv2.CHAIN_APPROX_SIMPLE - removes all redundant points and compress the contour
    # only two points in a square shape is stored
def get_contours(src_img):
    contours, hierarchy = cv2.findContours(src_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def draw_contours_1(blank, contours):
    contours = contours[0]
    for contour in contours:
        cv2.drawContours(blank, [contour], 0, (255, 255, 255), 1)
    return blank

def draw_contours_2(src_img, contour):
    cnt = contour[4]
    return cv2.drawContours(src_img, [cnt], -1, (255, 255, 255), 1)

def draw_contours_3(blank, contours):
    cv2.drawContours(blank, contours, -1, (255, 255, 255), 2)
    return blank


def just_thresh(src_img, blank):
    # show_image('thresh', src_img)
    show_plt_image(src_img)
    morph = morph_image(src_img)
    show_plt_image(morph)
    contours, hierarchy1 = get_contours(morph)
    print(len(contours))
    # contours = contours[0] if len(contours) == 2 else contours[1]
    mask = draw_contours_3(blank, contours)
    show_plt_image(mask)
    # resize1 = resize_image(mask)
    # x, y, w, h = cv2.boundingRect(contours1)
    # rectangle1 = cv2.rectangle(resize1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # show_image("thresh", mask)


def just_canny_edge(src_img, blank):
    edges = cv2.Canny(src_img, 150, 210)
    show_image("edges", edges)
    contours2, hierarchy2 = get_contours(edges)
    mask = draw_contours_3(blank, contours2)
    # resize2 = resize_image(result2)
    show_image("edges", mask)

def canny_and_thresh(src_img, blank):
    morph = morph_image(src_img)
    # show_image('morph', morph)
    edges_with_th = cv2.Canny(morph, 150, 210)
    show_image('canny edge', edges_with_th)
    contours3, hierarchy3 = get_contours(edges_with_th)
    mask = draw_contours_3(blank, contours3)
    # resize3 = resize_image(mask)
    # show_image("edges + thresh", resize3)
    show_plt_image(mask)


# img = cv2.imread('D:\Documents\Python\images\hand.jpg', 0)
img = cv2.imread('D:\Documents\Thesis\FSLRwithNLP\Datasets\Test_Images\R_8.jpg')
imgCopy = img.copy()
show_plt_image(img)

detector = HTM.HandDetector()
detected, pts_upper_left, pts_lower_right = detector.find_hands(imgCopy)
if detected:
    roi = img[pts_lower_right[1]:pts_upper_left[1], pts_upper_left[0]:pts_lower_right[0]]
    roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_CUBIC)
    # skin_masked = utils.skin_segmentation(roi)
    # show_image('skin', roi)
    show_plt_image(roi)
    # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # show_plt_image(th)
    # just_thresh(th, blank)
    # just_canny_edge(blur_img, blank)
    # canny_and_thresh(th, imgCopy)
else:
    print("nothing detected")

blank = np.zeros(img.shape, dtype='uint8')
# th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 6)

# _, th = cv2.threshold(blur_img, 155, 220, cv2.THRESH_BINARY)


