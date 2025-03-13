import cv2
import cv2.aruco as aruco
import mediapipe as mp
import threading
import argparse
import numpy as np
from queue import Queue
from pupil_detectors.detector_2d import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
import joblib
import pyautogui
import requests
import queue
import time

# Load Linear Regression Model (unchanged)
def load_linear_regression_model():
    try:
        lr_model = joblib.load('linearregressionmodelbucket3.joblib')
        print("Linear Regression model loaded successfully.")
        return lr_model
    except Exception as e:
        print(f"Failed to load Linear Regression model: {e}")
        return None

# FanLightController with session reuse
class FanLightController:
    def __init__(self, ip_address):
        self.base_url = f"http://{ip_address}"
        self.session = requests.Session()

    def set_speed(self, speed):
        if 0 <= speed <= 255:
            try:
                response = self.session.get(f"{self.base_url}/setspeed?speed={speed}", timeout=0.2)
                print(response.text)
            except requests.RequestException:
                pass  # Silently fail to avoid blocking
        else:
            print("Speed must be between 0 and 255")

    def set_light(self, brightness):
        if 0 <= brightness <= 255:
            try:
                response = self.session.get(f"{self.base_url}/setlight?brightness={brightness}", timeout=0.2)
                print(response.text)
            except requests.RequestException:
                pass
        else:
            print("Brightness must be between 0 and 255")

class EyeCamThread(threading.Thread):
    def __init__(self, stream_url, resolution, focal_length, gaze_queue):
        super().__init__()
        self.stream_url = stream_url
        self.resolution = tuple(resolution)
        self.gaze_queue = gaze_queue
        self.running = True
        self.detector_2d = Detector2D()
        self.camera = CameraModel(focal_length=focal_length, resolution=self.resolution)
        self.detector_3d = Detector3D(camera=self.camera, long_term_mode=DetectorMode.blocking)
        self.lr_model = load_linear_regression_model()
        self.frame_count = 0

    def run(self):
        cam = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)  # FFMPEG for MJPEG
        if not cam.isOpened():
            print(f"Error: Could not open stream {self.stream_url}")
            return

        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
        fps = 30  # Default FPS from mjpeg-streamer

        while self.running:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame from eye camera")
                time.sleep(0.01)  # Brief sleep to avoid tight loop
                continue

            self.frame_count += 1
            gaze_point = self.process_eye_frame(frame, self.frame_count, fps)
            if gaze_point is not None:
                try:
                    self.gaze_queue.put_nowait(gaze_point)
                except queue.Full:
                    pass  # Drop if queue is full

        cam.release()

    def process_eye_frame(self, frame, frame_number, fps):
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result_2d = self.detector_2d.detect(grayscale)
        result_2d["timestamp"] = frame_number / fps
        result_3d = self.detector_3d.update_and_detect(result_2d, grayscale)

        if result_3d['confidence'] > 0.5 and 'circle_3d' in result_3d and 'normal' in result_3d['circle_3d']:
            gaze_normal = result_3d['circle_3d']['normal']
            return self.predict_gaze_point(gaze_normal)
        return None

    def predict_gaze_point(self, gaze_normal):
        if self.lr_model is None:
            return None
        return tuple(map(int, self.lr_model.predict([gaze_normal])[0]))

    def stop(self):
        self.running = False

class FrontCamThread(threading.Thread):
    def __init__(self, stream_url, resolution, gaze_queue, controller):
        super().__init__()
        self.stream_url = stream_url
        self.resolution = tuple(resolution)
        self.gaze_queue = gaze_queue
        self.running = True
        self.controller = controller

        self.camera_matrix = np.array([[343.34511283, 0.0, 327.80111243],
                                      [0.0, 342.79698299, 231.06509007],
                                      [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([0, 0, 0, -0.001, -0.0], dtype=np.float32)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        self.aruco_params = aruco.DetectorParameters()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

        self.pinch_detected = False
        self.fan_on = False
        self.light_on = False
        self.fan_speeds = [150, 200, 120, 0]
        self.current_speed_index = 3
        self.thumb_tip_prev = None
        self.frame_count = 0

    def run(self):
        cam = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
        if not cam.isOpened():
            print(f"Error: Could not open stream {self.stream_url}")
            return

        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering

        while self.running:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame from front camera")
                time.sleep(0.01)
                continue

            self.frame_count += 1
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            corners, ids, _ = aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)
            results = self.hands.process(image_rgb)

            gaze_point = None
            try:
                gaze_point = self.gaze_queue.get_nowait()
            except queue.Empty:
                pass

            if gaze_point is not None and 0 <= gaze_point[0] < frame.shape[1] and 0 <= gaze_point[1] < frame.shape[0]:
                cv2.circle(frame, gaze_point, 15, (0, 0, 255), -1)
                if ids is not None:
                    for i, corner in enumerate(corners):
                        if self.point_in_polygon(gaze_point, corner[0]):
                            self.process_marker_action(ids[i][0], results)

            cv2.imshow('Front Camera', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

    def process_marker_action(self, marker_id, hand_results):
        if not hand_results.multi_hand_landmarks:
            return

        hand_landmarks = hand_results.multi_hand_landmarks[0]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))

        if distance < 0.035:
            if not self.pinch_detected:
                self.pinch_detected = True
                if marker_id == 2:
                    self.current_speed_index = (self.current_speed_index + 1) % len(self.fan_speeds)
                    self.controller.set_speed(self.fan_speeds[self.current_speed_index])
                elif marker_id == 3:
                    self.light_on = not self.light_on
                    self.controller.set_light(150 if self.light_on else 0)
            elif marker_id == 5 and self.thumb_tip_prev is not None:
                vertical_movement = thumb_tip.y - self.thumb_tip_prev
                if abs(vertical_movement) > 0.01:
                    scroll_amount = -vertical_movement * 400
                    pyautogui.press('volumeup' if scroll_amount > 0 else 'volumedown', presses=6)
            self.thumb_tip_prev = thumb_tip.y
        elif self.pinch_detected:
            self.pinch_detected = False

    def point_in_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def stop(self):
        self.running = False

def main(args):
    gaze_queue = Queue(maxsize=1)
    controller = FanLightController(args.device_ip)

    eye_cam_thread = EyeCamThread(args.eye_stream, args.eye_res, args.focal_length, gaze_queue)
    front_cam_thread = FrontCamThread(args.front_stream, args.front_res, gaze_queue, controller)

    eye_cam_thread.start()
    front_cam_thread.start()

    try:
        eye_cam_thread.join()
        front_cam_thread.join()
    except KeyboardInterrupt:
        print("Stopping threads...")
        eye_cam_thread.stop()
        front_cam_thread.stop()
        eye_cam_thread.join()
        front_cam_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated Eye and Hand Control System")
    parser.add_argument("--eye_stream", type=str, default="http://192.168.195.53:8081/?action=stream", help="Eye camera stream URL")
    parser.add_argument("--front_stream", type=str, default="http://192.168.195.53:8080/?action=stream", help="Front camera stream URL")
    parser.add_argument("--eye_res", nargs=2, type=int, default=[320, 240], help="Eye camera resolution")
    parser.add_argument("--front_res", nargs=2, type=int, default=[640, 480], help="Front camera resolution")
    parser.add_argument("--focal_length", type=float, default=84, help="Focal length of the eye camera")
    parser.add_argument("--device_ip", type=str, default="192.168.195.148", help="IP address of the fan/light controller")
    args = parser.parse_args()
    
    main(args)