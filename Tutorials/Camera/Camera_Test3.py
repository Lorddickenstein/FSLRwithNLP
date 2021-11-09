import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
cap_size = (int(cap.get(3)), int(cap.get(4)))
out = cv.VideoWriter('output.avi', fourcc, 20.0, cap_size)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.flip(frame, 0)
    # write the flipped frame
    # out.write(frame)
    cv.imshow('frame', frame)
    print(frame.ndim)
    if cv.waitKey(1) == ord('q'):
        break
# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()