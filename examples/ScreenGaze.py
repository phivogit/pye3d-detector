import cv2
import threading
import argparse
import numpy as np
from queue import Queue
from pupil_detectors.detector_2d import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
import joblib
import queue
import pyautogui
import time

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
    def __init__(self, cam_id, resolution, focal_length, gaze_queue, front_cam_thread):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.resolution = resolution
        self.gaze_queue = gaze_queue
        self.running = True
        self.front_cam_thread = front_cam_thread

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
        return prediction

    def stop(self):
        self.running = False

class FrontCamThread(threading.Thread):
    def __init__(self, cam_id, resolution, gaze_queue):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.resolution = resolution
        self.gaze_queue = gaze_queue
        self.running = True

        self.camera_matrix = np.array([[343.34511283, 0.0, 327.80111243],
                                       [0.0, 342.79698299, 231.06509007],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([0, 0, 0, -0.001, -0.0])
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.screen_coords = None

    def run(self):
        cam = cv2.VideoCapture(self.cam_id)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        while self.running:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame from front camera")
                break

            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

            # Detect ArUco markers
            corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_parameters)

            if ids is not None and len(ids) == 4:
                # Get the screen coordinates from the ArUco markers
                self.screen_coords = self.get_screen_coords(corners, ids)
                if self.screen_coords is not None:
                    print("Perspective transform matrix computed")

            try:
                gaze_point = self.gaze_queue.get_nowait()
                if self.screen_coords is not None:
                    # Convert gaze point to numpy array
                    gaze_point = np.array([[gaze_point[0], gaze_point[1]]], dtype=np.float32)

                    # Apply perspective transform
                    transformed_point = cv2.perspectiveTransform(gaze_point.reshape(-1, 1, 2), self.screen_coords)
                    screen_x, screen_y = transformed_point[0][0]

                    # Check if the point is within the screen bounds
                    if 0 <= screen_x <= 1 and 0 <= screen_y <= 1:
                        print(f"Gaze point: ({gaze_point[0][0]}, {gaze_point[0][1]}), "
                              f"Transformed: ({screen_x}, {screen_y})")
                        
                        # Map the normalized coordinates to screen resolution
                        screen_width, screen_height = 1920, 1080
                        mouse_x = int(screen_x * screen_width)
                        mouse_y = int(screen_y * screen_height)
                        
                        pyautogui.moveTo(mouse_x, mouse_y)
                        
                        # For visualization
                        frame_points = cv2.perspectiveTransform(np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float32), np.linalg.inv(self.screen_coords))
                        cv2.polylines(frame, [frame_points.astype(int)], True, (0, 255, 0), 2)
                        
                        # Draw the ArUco markers
                        for corner in corners:
                            cv2.polylines(frame, [corner.astype(int)], True, (0, 0, 255), 2)
                        
                        # Draw the gaze point
                        cv2.circle(frame, (int(gaze_point[0][0]), int(gaze_point[0][1])), 5, (255, 0, 0), -1)
                        
            except queue.Empty:
                pass

            cv2.imshow("Front Camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

    def get_screen_coords(self, corners, ids):
        if len(corners) != 4:
            print(f"Detected {len(corners)} ArUco markers, expected 4.")
            return None

        # Initialize sorted_corners with None
        sorted_corners = [None, None, None, None]

        # Sort corners based on their IDs
        for corner, id in zip(corners, ids):
            if id[0] == 0:
                sorted_corners[0] = corner
            elif id[0] == 2:
                sorted_corners[1] = corner
            elif id[0] == 3:
                sorted_corners[2] = corner
            elif id[0] == 1:
                sorted_corners[3] = corner

        # Check if all markers were detected
        if any(corner is None for corner in sorted_corners):
            print("Not all required ArUco markers were detected.")
            return None

        # Get the outer corners of each marker
        tl = sorted_corners[0][0][0]  # Top-left corner of top-left marker (ID 0)
        tr = sorted_corners[1][0][1]  # Top-right corner of top-right marker (ID 2)
        br = sorted_corners[2][0][2]  # Bottom-right corner of bottom-right marker (ID 3)
        bl = sorted_corners[3][0][3]  # Bottom-left corner of bottom-left marker (ID 1)

        # Define the quadrilateral
        quad = np.array([tl, tr, br, bl], dtype=np.float32)

        # Define the actual screen coordinates of the markers
        screen_width, screen_height = 1920, 1080
        actual_corners = np.array([
            [81, 51],              # Top-left (ID 0)
            [1838, 51],            # Top-right (ID 2)
            [1838, 1036],          # Bottom-right (ID 3)
            [81, 1036]             # Bottom-left (ID 1)
        ], dtype=np.float32)

        # Normalize the actual corners to [0, 1] range
        actual_corners[:, 0] /= screen_width
        actual_corners[:, 1] /= screen_height

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(quad, actual_corners)

        return M

    def stop(self):
        self.running = False

def main(args):
    gaze_queue = Queue(maxsize=5)

    eye_cam_thread = EyeCamThread(args.eye_cam, args.eye_res, args.focal_length, gaze_queue, None)
    front_cam_thread = FrontCamThread(args.front_cam, args.front_res, gaze_queue)
    eye_cam_thread.front_cam_thread = front_cam_thread

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
    parser = argparse.ArgumentParser(description="Dual camera eye tracking system")
    parser.add_argument("--eye_cam", type=int, default=2, help="Eye camera index")
    parser.add_argument("--front_cam", type=int, default=1, help="Front camera index")
    parser.add_argument("--eye_res", nargs=2, type=int, default=[320, 240], help="Eye camera resolution")
    parser.add_argument("--front_res", nargs=2, type=int, default=[640, 480], help="Front camera resolution")
    parser.add_argument("--focal_length", type=float, default=184.7, help="Focal length of the eye camera")
    args = parser.parse_args()
    
    main(args) 