import cv2
import time
import facial_tracking.conf as conf

from facial_tracking.faceMesh import FaceMesh
from facial_tracking.eye import Eye
from facial_tracking.lips import Lips


class FacialTracker:
    """
    Tracks facial expressions, including eye closure and yawning.
    """

    def __init__(self):
        self.fm = FaceMesh()
        self.left_eye = None
        self.right_eye = None
        self.lips = None
        self.left_eye_closed_frames = 0
        self.right_eye_closed_frames = 0
        self.eyes_closed = False
        self.yawn_detected = False
        self.detected = False  # Flag to check if a face is detected

    def process_frame(self, frame):
        """Processes the frame and detects eye closure and yawning."""
        self.detected = False
        self.fm.process_frame(frame)
        self.fm.draw_mesh_lips()

        if self.fm.mesh_result.multi_face_landmarks:
            self.detected = True
            for face_landmarks in self.fm.mesh_result.multi_face_landmarks:
                self.left_eye = Eye(frame, face_landmarks, conf.LEFT_EYE)
                self.right_eye = Eye(frame, face_landmarks, conf.RIGHT_EYE)
                self.lips = Lips(frame, face_landmarks, conf.LIPS)

                self._check_eyes_status()
                self._check_yawn_status()

    def _check_eyes_status(self):
        """Checks if eyes are closed and updates status."""
        self.eyes_closed = False

        if self.left_eye.eye_closed():
            self.left_eye_closed_frames += 1
        else:
            self.left_eye_closed_frames = 0
            self.left_eye.iris.draw_iris(True)

        if self.right_eye.eye_closed():
            self.right_eye_closed_frames += 1
        else:
            self.right_eye_closed_frames = 0
            self.right_eye.iris.draw_iris(True)

        # Update eye closure status
        if self._left_eye_closed() or self._right_eye_closed():
            self.eyes_closed = True

    def _check_yawn_status(self):
        """Checks if the driver is yawning."""
        self.yawn_detected = self.lips.mouth_open()

    def _left_eye_closed(self, threshold=conf.FRAME_CLOSED):
        """Checks if the left eye is closed for more than the threshold frames."""
        return self.left_eye_closed_frames > threshold

    def _right_eye_closed(self, threshold=conf.FRAME_CLOSED):
        """Checks if the right eye is closed for more than the threshold frames."""
        return self.right_eye_closed_frames > threshold


def main():
    cap = cv2.VideoCapture(conf.CAM_ID)
    cap.set(3, conf.FRAME_W)
    cap.set(4, conf.FRAME_H)
    facial_tracker = FacialTracker()
    ptime = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        facial_tracker.process_frame(frame)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'FPS: {int(fps)}', (30, 30), 0, 0.6, conf.TEXT_COLOR, 1, lineType=cv2.LINE_AA)

        if facial_tracker.detected:
            cv2.putText(frame, 'Eyes Closed' if facial_tracker.eyes_closed else 'Eyes Open', 
                        (30, 70), 0, 0.8, conf.WARN_COLOR, 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, 'Yawning' if facial_tracker.yawn_detected else 'No Yawn', 
                        (30, 110), 0, 0.8, conf.WARN_COLOR, 2, lineType=cv2.LINE_AA)

        cv2.imshow('Facial Tracking', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
