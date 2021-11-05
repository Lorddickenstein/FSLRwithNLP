import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2
import Application.HandTrackingModule as HTM
import Application.utils as utils
import os
import shutil

def detect_blur_fft(image, size=80, thresh=10, vis=False):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)

    # check to see if we are visualizing our output
    if vis:
        # compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs(fft_shift))

        # display the original input image
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # display the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        # show our plots
        plt.show()
    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fft_shift[cY - size:cY + size, cX - size:cX + size] = 0
    fft_shift = np.fft.ifftshift(fft_shift)
    recon = np.fft.ifft2(fft_shift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return (mean, mean <= thresh)

def static_image(test=False):
    # load the input image from disk, resize it, and convert it to grayscale
    # orig = cv2.imread('D:\Documents\Thesis\FSLRwithNLP\Tutorials\Camera\keyframes\keyframe21.jpg')
    orig = cv2.imread('D:\Documents\Thesis\FSLRwithNLP\Tutorials\Camera\keyframes\cropped images\\1_F_1455.30.jpg')
    orig = imutils.resize(orig, width=500)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    # apply our blur detector using the FFT
    (mean, blurry) = detect_blur_fft(gray, size=60,
        thresh=19.5, vis=True)

    # draw on the image, indicating whether or not it is blurry
    image = np.dstack([gray] * 3)
    color = (0, 0, 255) if blurry else (0, 255, 0)
    text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
    text = text.format(mean)
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                color, 2)
    print("[INFO] {}".format(text))
    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

    if test:
        for radius in range(1, 30, 2):
            image = gray.copy()
            if radius > 0:
                image = cv2.GaussianBlur(image, (radius, radius), 0)
                (mean, blurry) = detect_blur_fft(image, size=30, thresh=20, vis=False)
                image = np.dstack([image] * 3)
                color = (0, 0, 255) if blurry else (0, 255, 0)
                text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
                text = text.format(mean)
                cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)
                print("[INFO] Kernel: {}, Result: {}".format(radius, text))
                # show the image
            cv2.imshow("Test Image", image)
            cv2.waitKey(10)

def video_stream():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _, frame = cap.read()
        height, width, channel = frame.shape
        if not _:
            break
        frame = imutils.resize(frame, width=1000)
        frameCopy = frame.copy()

        detected, pts_upper_left, pts_lower_right = detector.find_hands(frameCopy)

        if detected:
            # cv2.rectangle(frame, pts_upper_left, pts_lower_right, (255, 0, 0), 3)
            try:
                roi = frameCopy[pts_lower_right[1]:pts_upper_left[1], pts_upper_left[0]:pts_lower_right[0]]
                skin = utils.skin_segmentation(roi)
                skin = utils.resize_image(skin, height=120, width=120)
                cv2.imshow('Cropped', skin)
                gray = imutils.resize(skin, width=500)
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                (mean, blurry) = detect_blur_fft(gray, size=60, thresh=9.2, vis=False)
                color = (0, 0, 255) if blurry else (0, 255, 0)
                text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
                text = text.format(mean)
                cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)
                if is_capturing:
                    name = str(i) + "_" + text + ".jpg"
                    cv2.imwrite(os.path.join(save_crop, name), skin)
                    # if 'Not' in text:
                    #     cv2.imwrite(os.path.join(save_crop, name), gray)
                    cv2.imwrite(os.path.join(save_path, name), frameCopy)
                    i += 1

            except Exception:
                pass
        else:
            cv2.putText(frame, "No hands detected...", (10, height - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q") or key == 27:
            break
        elif key == ord("s"):
            try:
                shutil.rmtree(save_path)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
            os.makedirs(save_path)
            os.makedirs(save_crop)

            is_capturing = True
            i = 0
        elif key == ord("e"):
            is_capturing = False
    # do a bit of cleanup
    cap.release()
    cv2.destroyAllWindows()

save_path = 'D:\Documents\Thesis\FSLRwithNLP\Tutorials\Camera\\trash'
save_crop = 'D:\Documents\Thesis\FSLRwithNLP\Tutorials\Camera\\trash\cropped'
is_capturing = False
detector = HTM.HandDetector()
# static_image()
video_stream()