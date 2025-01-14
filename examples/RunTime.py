import cv2
import cv2.aruco as aruco
import threading
import argparse
import numpy as np
from queue import Queue
from pupil_detectors.detector_2d import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
import joblib
import queue
import mediapipe as mp
import requests
import time
#chay tat mo den
def load_mixed_model():
    try:
        lr_model = joblib.load('linear_regression_model.joblib')
        dt_model = joblib.load('decision_tree_model.joblib')
        nn_model = joblib.load('nearest_neighbors_model.joblib')
        
        with open('threshold.txt', 'r') as f:
            threshold = float(f.read())
        
        def predict_mixed_model(gaze_direction):
            dist, _ = nn_model.kneighbors([gaze_direction])
            if dist < threshold:
                return dt_model.predict([gaze_direction])[0]
            else:
                return lr_model.predict([gaze_direction])[0]
        
        print("Pre-trained mixed model loaded successfully.")
        return predict_mixed_model
    except Exception as e:
        print(f"Failed to load pre-trained model: {e}")
        return None

class EyeCamThread(threading.Thread):
    def __init__(self, cam_id, resolution, focal_length, gaze_queue):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.resolution = resolution
        self.gaze_queue = gaze_queue
        self.running = True

        self.detector_2d = Detector2D()
        self.camera = CameraModel(focal_length=focal_length, resolution=resolution)
        self.detector_3d = Detector3D(camera=self.camera, long_term_mode=DetectorMode.blocking)
        self.mixed_model = load_mixed_model()

    def run(self):
        cam = cv2.VideoCapture(self.cam_id)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        fps = 30
        frame_count = 0

        while self.running:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame from eye camera")
                break

            if frame_count % 2 == 0:  # Process every other frame
                gaze_point = self.process_eye_frame(frame, frame_count, fps)
                if gaze_point is not None:
                    self.gaze_queue.put(gaze_point)

            frame_count += 1

        cam.release()

    def process_eye_frame(self, frame, frame_number, fps):
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result_2d = self.detector_2d.detect(grayscale)
        result_2d["timestamp"] = frame_number / fps
        result_3d = self.detector_3d.update_and_detect(result_2d, grayscale)
        
        if result_3d['confidence'] > 0.6 and 'circle_3d' in result_3d and 'normal' in result_3d['circle_3d']:
            gaze_normal = result_3d['circle_3d']['normal']
            return self.predict_gaze_point(gaze_normal)
        return None

    def predict_gaze_point(self, gaze_normal):
        if self.mixed_model is None:
            return None
        prediction = self.mixed_model(gaze_normal)
        return tuple(map(int, prediction))

    def stop(self):
        self.running = False

ESP32_IP = "192.168.1.87"  # Replace with your ESP32's IP address
ESP32_URL = f"http://{ESP32_IP}/control"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)

def send_command(channel, angle):
    try:
        payload = {"channel": channel, "angle": angle}
        response = requests.post(ESP32_URL, json=payload)
        print(f"ESP32 response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with ESP32: {e}")

def control_servos(channel, angle):
    send_command(channel, angle)
    

class FrontCamThread(threading.Thread):
    def __init__(self, cam_id, resolution, gaze_queue):
        super().__init__() 
        self.cam_id = cam_id
        self.resolution = resolution
        self.gaze_queue = gaze_queue
        self.running = True
        self.pinch_detected = False
        self.looking_at_id2 = False
        self.looking_at_id5 = False
        self.last_command_time = 0
        self.command_cooldown = 1  # 1 second cooldown between commands

        self.camera_matrix = np.array([[343.34511283, 0.0, 327.80111243],
                                       [0.0, 342.79698299, 231.06509007],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([0, 0, 0, -0.001, -0.0])

        # ArUco dictionary
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        self.aruco_params = aruco.DetectorParameters()
    
    
    def expand_marker_area(self, corners, expand_factor=1.5):
        """Expand the marker area by a given factor."""
        center = np.mean(corners, axis=0)
        expanded_corners = []
        for corner in corners:
            vector = corner - center
            expanded_corner = center + vector * expand_factor
            expanded_corners.append(expanded_corner)
        return np.array(expanded_corners)
    
    def run(self):
        cam = cv2.VideoCapture(self.cam_id)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        gaze_point = None

        while self.running:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame from front camera")
                break

            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

            # Detect ArUco markers
            corners, ids, _ = aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)
            
            # Process hand tracking
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(image_rgb)

            try:
                gaze_point = self.gaze_queue.get_nowait()
            except queue.Empty:
                pass

# At the beginning of the frame processing loop in the run method:
            self.looking_at_id2 = False
            self.looking_at_id5 = False

            if gaze_point is not None:
                if 0 <= gaze_point[0] < frame.shape[1] and 0 <= gaze_point[1] < frame.shape[0]:
                    cv2.circle(frame, gaze_point, 15, (0, 0, 255), 5)
                
                # Check if gaze point is on any detected marker
                if ids is not None:
                    for i, corner in enumerate(corners):
                        expanded_corner = self.expand_marker_area(corner[0])
                        if self.point_in_polygon(gaze_point, expanded_corner):
                            marker_id = ids[i][0]
                            if marker_id == 2:
                                self.looking_at_id2 = True
                            elif marker_id == 5:
                                self.looking_at_id5 = True
            # Process hand landmarks
            if hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))

                if distance < 0.035:
                    self.pinch_detected = True
                else:
                    self.pinch_detected = False

            # Control servos based on gaze and pinch
            current_time = time.time()
            if self.pinch_detected and (current_time - self.last_command_time) > self.command_cooldown:
                    if self.looking_at_id2:
                        control_servos(0, 45)
                        self.last_command_time = current_time
                        print("Looking at id2 and pinching: Servos set to 0 and 45 degrees")
                    elif self.looking_at_id5:
                        control_servos(15, 45)
                        self.last_command_time = current_time
                        print("Looking at id5 and pinching: Servos set to 15 and 45 degrees")
                    else:
                        pass 
            cv2.imshow("Front Camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                break

        cam.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False

    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon."""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

def main(args):
    gaze_queue = Queue(maxsize=1)

    eye_cam_thread = EyeCamThread(args.eye_cam, args.eye_res, args.focal_length, gaze_queue)
    front_cam_thread = FrontCamThread(args.front_cam, args.front_res, gaze_queue)

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
    parser = argparse.ArgumentParser(description="Dual camera eye tracking system with WiFi servo control")
    parser.add_argument("--eye_cam", type=int, default=2, help="Eye camera index")
    parser.add_argument("--front_cam", type=int, default=1, help="Front camera index")
    parser.add_argument("--eye_res", nargs=2, type=int, default=[320, 240], help="Eye camera resolution")
    parser.add_argument("--front_res", nargs=2, type=int, default=[640, 480], help="Front camera resolution")
    parser.add_argument("--focal_length", type=float, default=184.7, help="Focal length of the eye camera")
    args = parser.parse_args()
    
    main(args)