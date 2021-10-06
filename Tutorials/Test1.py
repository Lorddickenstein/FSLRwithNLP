# Test if HandTrackingModule is working
import os
import cv2
import Application.HandTrackingModule as HTM

def show_image(name, img):
    # plt.imshow(img, cmap='gray')
    # plt.show()
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

path = "D:\Documents\Thesis\FSLRwithNLP\Datasets\Test_Images"
file_name = "L2.jpg"
img = cv2.imread(os.path.join(path, file_name))
show_image('grayscale', img)

detector = HTM.HandDetector()
detected, pts_upper_left, pts_lower_right = detector.find_hands(img)

if detected:
    print("hey")
else:
    print("Hello")
