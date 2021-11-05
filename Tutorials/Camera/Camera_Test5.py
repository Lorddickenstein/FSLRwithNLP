import cv2
import numpy as np
import matplotlib.pyplot as plt
import Application.utils as utils
import Application.HandTrackingModule as HTM
import Application.SignClassificationModule as SCM
import imutils
import os

# Open the camera
cap = cv2.VideoCapture(0)
detector = HTM.HandDetector()

text = 'Not Capturing'
is_capturing = False

frm_num_arr = []
frm_gradients = []
prevGradient = np.array([])
frm_arr = []

frm_num = 0
figures_path = 'D:\Documents\Thesis\FSLRwithNLP\Tutorials\Camera\Figures'
gradient_thresh_value = 3.1
TEN_MILLION = 10000000.0

while cap.isOpened():
    _, frame = cap.read()
    # Filter lines to make it sharper and smoother
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = imutils.resize(frame, width=1000)
    height, width, channel = frame.shape
    frameCopy = frame.copy()

    if not _:
        print("Ignoring empty camera frame.")
        continue;

    cv2.putText(frame, text, (10, int(0.98 * height)), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    if is_capturing:
        detected, pts_upper_left, pts_lower_right = detector.find_hands(frameCopy)
        if detected:
            roi = frameCopy[pts_lower_right[1]:pts_upper_left[1], pts_upper_left[0]:pts_lower_right[0]]

            try:
                currFrame = utils.convert_to_grayscale(frameCopy)
                # currFrame = utils.resize_image(currFrame, height=120, width=120)
                sobelx = cv2.Sobel(currFrame, cv2.CV_64F, 1, 0, ksize=cv2.FILTER_SCHARR)
                sobely = cv2.Sobel(currFrame, cv2.CV_64F, 0, 1, ksize=cv2.FILTER_SCHARR)
                currGradient = np.sqrt(np.square(sobelx) + np.square(sobely))

                if frm_num != 0:
                    frm_diff = cv2.absdiff(currGradient, prevGradient)
                    frm_sum = cv2.sumElems(frm_diff)
                    frm_sum = frm_sum[0]/TEN_MILLION
                    if frm_sum < gradient_thresh_value:
                        frm_sum = gradient_thresh_value
                    print(frm_sum, frm_num)

                    frm_gradients.append(frm_sum)
                    frm_num_arr.append(frm_num)
                    frm_arr.append(frame)

                cv2.rectangle(frame, pts_upper_left, pts_lower_right, (255, 0, 0), 3)
                prevGradient = currGradient
                frm_num += 1
            except Exception as exc:
                pass

    cv2.imshow('Original', frame)
    key = cv2.waitKey(5) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord('s'):
        if not is_capturing:
            text = 'Capturing'
            is_capturing = True
    elif key == ord('e'):
        if is_capturing:
            plt.plot(frm_num_arr, frm_gradients)
            plt.savefig(os.path.join(figures_path, 'Figures.png'), bbox_inches='tight')
            plt.show()

            # for frm_gradient in frm_gradients:
            #     if frm_gradient == gradient_thresh_value:

            frm_num_arr = []
            frm_gradients = []
            frm_arr = []
            frm_num = 0

            text = 'Not Capturing'
            is_capturing = False


cap.release()
cv2.destroyAllWindows()