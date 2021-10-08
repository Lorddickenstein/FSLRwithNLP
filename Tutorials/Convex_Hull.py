import cv2
import numpy as np

img = cv2.imread('D:\Documents\Thesis\FSLRwithNLP\Datasets\Test_Images\C.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
