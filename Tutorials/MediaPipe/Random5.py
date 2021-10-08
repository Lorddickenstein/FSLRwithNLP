import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# img = cv2.imread('D:\Documents\Thesis\OurDataset\Raw Dataset\A\color_0_1.jpg')
img = cv2.imread('D:\Documents\Thesis\FSLRwithNLP\Datasets\Test_Images\R_6.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img.flags.writeable = False
results = hands.process(img)
img.flags.writeable = True
blank = np.zeros(img.shape, dtype='uint8')
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(blank, hand_landmarks, mp_hands.HAND_CONNECTIONS)
else:
    print("nothing detected")

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blank), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.show()