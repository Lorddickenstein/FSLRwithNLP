import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionConf=0.5, trackingConf=0.5):
        # Arguments
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf

        # Initialize mediapipe variables
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionConf, self.trackingConf)
        self.mpDraw = mp.solutions.drawing_utils


    def find_hands(self, src_img):
        # Convert frame to rgb for mediapipe
        rgb = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        rgb.flags.writeable = False
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLMs in results.multi_hand_landmarks:
                # left_x_pts, left_y_pts, right_x_pts, right_y_pts = [], [], [], []

                for id, lm in enumerate(handLMs.landmark):
                    continue
                    # if handedness.classification[0].label == 'Right':
                    #     height, width, channel = src_img.shape
                    #     cx, cy = int(lm.x * width), int(lm.y * height)
                    #     right_x_pts.append(cx)
                    #     right_y_pts.append(cy)
                    #
                    # if handedness.classification[0].label == 'left':
                    #     height, width, channel = src_img.shape
                    #     cx, cy = int(lm.x * width), int(lm.y * height)
                    #     left_x_pts.append(cx)
                    #     left_y_pts.append(cy)

                # Find the right maximum and minimum xy values
                right_x_max, right_x_min = max(right_x_pts), min(right_x_pts),
                right_y_max, right_y_min = max(right_y_pts), min(right_y_pts)

                # Find the left maximum and minimum y values
                left_x_max, left_x_min = max(left_x_pts), min(left_x_pts),
                left_y_max, left_y_min = max(left_y_pts), min(left_y_pts)

                self.mpDraw.draw_landmarks(src_img, handLMs, self.mpHands.HAND_CONNECTIONS)
            # for idx, handedness in enumerate(results.multi_handedness):

        return src_img

                # x_pts = []
                # y_pts = []
                # for id, lm in enumerate(handLMs.landmark):
                #     h, w, c = src_img.shape
                #     cx, cy = int(lm.x * w), int(lm.y * h)
                #     x_pts.append(cx)
                #     y_pts.append(cy)
                #
                # # Find the max and min points
                # y_max, y_min, x_max, x_min = max(y_pts), min(y_pts), max(x_pts), min(x_pts)
                # cv2.rectangle(src_img, (x_min - 20, y_max + 20), (x_max + 20, y_min - 20), (255, 0, 0), 3)
                #
                # self.mp_draw.draw_landmarks(src_img, handLMs, self.mpHands.HAND_CONNECTIONS)

def main():
    # Open the camera
    cap = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0

    detector = HandDetector()
    while True:
        _, frame = cap.read()
        if not _:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image
        flip = cv2.flip(frame, 1)
        frame = detector.find_hands(flip)

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # Show Fps
        cv2.putText(frame, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('Original', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
