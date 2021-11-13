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
        self.extra = 0.08


    def find_hands(self, src_img, draw=False):
        # Convert frame to rgb for mediapipe
        rgb = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        rgb.flags.writeable = False
        self.results = self.hands.process(rgb)

        if self.results.multi_hand_landmarks:
            x_pts, y_pts = [], []
            i = 0
            for handLMs in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLMs.landmark):
                    h, w, c = src_img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    x_pts.append(cx)
                    y_pts.append(cy)
                    i += 1
                if draw:
                    self.mpDraw.draw_landmarks(src_img, handLMs, self.mpHands.HAND_CONNECTIONS)

            if i == 21 or i == 42:
                # Find the max and min points
                y_max, y_min, x_max, x_min = max(y_pts), min(y_pts), max(x_pts), min(x_pts)
                extra_space = src_img.shape[0] * self.extra if src_img.shape[0] < src_img.shape[1] else src_img.shape[1] * self.extra
                tuple_upper_left = (int(x_min - extra_space), int(y_max + extra_space))
                tuple_lower_right = (int(x_max + extra_space), int(y_min - extra_space))
                x1, y1 = (value if value > 0 else 0 for value in tuple_upper_left)
                x2, y2 = (value if value > 0 else 0 for value in tuple_lower_right)
                return True, (x1, y1), (x2, y2)
        return False, (), ()

def main():
    # Open the camera
    cap = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0

    detector = HandDetector()
    while True:
        _, frame = cap.read()
        height, width, channel = frame.shape
        if not _:
            print("Ignoring empty camera frame.")
            continue

        detected, pts_upper_left, pts_lower_right = detector.find_hands(frame)

        if detected:
            cv2.rectangle(frame, pts_upper_left, pts_lower_right, (255, 0, 0), 3)
        else:
            cv2.putText(frame, "No hands detected...", (10, height - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # Show Fps
        cv2.putText(frame, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN , 3, (0, 255, 0), 3)

        cv2.imshow('Original', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
