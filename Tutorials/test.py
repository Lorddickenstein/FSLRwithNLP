import os
from datetime import datetime
import cv2
import numpy as np

blank = np.zeros((500, 1200), dtype='uint8')
# (img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None)
Fonts = ['FONT_HERSHEY_SIMPLEX ', 'FONT_HERSHEY_PLAIN', 'FONT_HERSHEY_DUPLEX', 'FONT_HERSHEY_COMPLEX ',
         'FONT_HERSHEY_TRIPLEX', 'FONT_HERSHEY_COMPLEX_SMALL', 'FONT_HERSHEY_SCRIPT_SIMPLEX ', 'FONT_HERSHEY_SCRIPT_COMPLEX ']

color = (255, 255, 0, 128)

cv2.putText(blank, Fonts[0], (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
cv2.putText(blank, Fonts[1], (30, 100), cv2.FONT_HERSHEY_PLAIN, 1, color, 3, cv2.LINE_AA)
cv2.putText(blank, Fonts[2], (30, 150), cv2.FONT_HERSHEY_DUPLEX, 1, color, 3, cv2.LINE_AA)
cv2.putText(blank, Fonts[3], (30, 200), cv2.FONT_HERSHEY_COMPLEX, 1, color, 3, cv2.LINE_AA)
cv2.putText(blank, Fonts[4], (30, 250), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 3, cv2.LINE_AA)
cv2.putText(blank, Fonts[5], (30, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2, cv2.LINE_AA)
cv2.putText(blank, Fonts[6], (30, 350), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, color, 3, cv2.LINE_AA)
cv2.putText(blank, Fonts[6], (30, 400), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, color, 3, cv2.LINE_AA)

cv2.imshow('Fonts', blank)
cv2.waitKey(0)
cv2.destroyAllWindows()
