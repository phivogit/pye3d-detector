import cv2
import threading
import argparse
import numpy as np
import pyautogui
from pupil_detectors.detector_2d import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
import joblib

class SharedGazeData:
    def __init__(self):
        self.gaze_point = None
        self.lock = threading.Lock()

    def update(self, gaze_point):
        with self.lock:
            self.gaze_point = gaze_point

    def get(self):
        with self.lock:
            return self.gaze_point

class CamThread(threading.Thread):
    def __init__(self, preview_name, cam_id, resolution, is_eye_cam=False, focal_length=None, 
                 shared_gaze_data=None, camera_matrix=None, dist_coeffs=None, lr_model=None):
        threading.Thread.__init__(self)
        self.preview_name = preview_name
        self.cam_id = cam_id
        self.resolution = resolution
        self.is_eye_cam = is_eye_cam
        self.shared_gaze_data = shared_gaze_data
        self.running = True
        self.debug_info = ""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.lr_model = lr_model
        self.screen_transform = None
        
        # ArUco marker detection for screen coordinate mapping
        if not is_eye_cam:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
            self.aruco_parameters = cv2.aruco.DetectorParameters()

        if is_eye_cam:
            self.detector_2d = Detector2D()
            self.camera = CameraModel(focal_length=focal_length, resolution=resolution)
            self.detector_3d = Detector3D(camera=self.camera, long_term_mode=DetectorMode.blocking)

    def run(self):
        print(f'Starting {self.preview_name}')
        self.cam_preview()

    def stop(self):
        self.running = False

    def process_eye_frame(self, frame, frame_number, fps):
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result_2d = self.detector_2d.detect(grayscale)
        result_2d["timestamp"] = frame_number / fps
        result_3d = self.detector_3d.update_and_detect(result_2d, grayscale)
        
        if result_3d['confidence'] > 0.6 and 'circle_3d' in result_3d and 'normal' in result_3d['circle_3d']:
            gaze_normal = result_3d['circle_3d']['normal']
            gaze_point = self.predict_gaze_point(gaze_normal)
            if gaze_point is not None:
                self.shared_gaze_data.update(gaze_point)
                self.debug_info = f"Predicted gaze point: {gaze_point}"
            else:
                self.debug_info = "Invalid gaze prediction"
        else:
            self.debug_info = "Low confidence or missing gaze data"
        
        return result_3d

    def predict_gaze_point(self, gaze_normal):
        if self.lr_model is None:
            return None
        
        prediction = self.lr_model.predict([gaze_normal])[0]
        return tuple(map(int, prediction))

    def get_screen_coords(self, corners, ids):
        if len(corners) != 4:
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
            return None

        # Get the outer corners of each marker
        tl = sorted_corners[0][0][0]  # Top-left corner of top-left marker (ID 0)
        tr = sorted_corners[1][0][1]  # Top-right corner of top-right marker (ID 2)
        br = sorted_corners[2][0][2]  # Bottom-right corner of bottom-right marker (ID 3)
        bl = sorted_corners[3][0][3]  # Bottom-left corner of bottom-left marker (ID 1)

        # Define the quadrilateral
        quad = np.array([tl, tr, br, bl], dtype=np.float32)

        # Define the actual screen coordinates of the markers
        screen_width, screen_height = pyautogui.size()
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

    def visualize_eye_result(self, frame, result_3d):
        if 'ellipse' in result_3d:
            ellipse = result_3d["ellipse"]
            cv2.ellipse(frame, 
                        tuple(int(v) for v in ellipse["center"]),
                        tuple(int(v / 2) for v in ellipse["axes"]),
                        ellipse["angle"], 0, 360, (0, 255, 0), 2)
        return frame

    def map_gaze_to_screen(self, gaze_point):
        if self.screen_transform is None:
            return None
            
        # Convert gaze point to the format expected by perspectiveTransform
        point = np.array([[[gaze_point[0], gaze_point[1]]]], dtype=np.float32)
        
        try:
            # Apply perspective transformation
            transformed_point = cv2.perspectiveTransform(point, self.screen_transform)
            screen_x, screen_y = transformed_point[0][0]
            
            # Check if the point is within screen bounds (normalized coordinates)
            if 0 <= screen_x <= 1 and 0 <= screen_y <= 1:
                # Map normalized coordinates to screen resolution
                screen_width, screen_height = pyautogui.size()
                mouse_x = int(screen_x * screen_width)
                mouse_y = int(screen_y * screen_height)
                return (mouse_x, mouse_y)
        except Exception as e:
            print(f"Error mapping gaze to screen: {e}")
            
        return None

    def cam_preview(self):
        cam = cv2.VideoCapture(self.cam_id)
        if not cam.isOpened():
            print(f"Error: Could not open camera {self.cam_id}")
            return

        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Disable autofocus for the front camera
        if not self.is_eye_cam:
            cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0 = disable autofocus

        fps = 30
        frame_count = 0

        while self.running:
            ret, frame = cam.read()
            if not ret:
                print(f"Failed to grab frame from {self.preview_name}")
                break

            if self.is_eye_cam:
                result_3d = self.process_eye_frame(frame, frame_count, fps)
                frame = self.visualize_eye_result(frame, result_3d)
            else:
                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
                
                # Detect ArUco markers for screen coordinate mapping
                corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_parameters)
                if ids is not None and len(ids) == 4:
                    self.screen_transform = self.get_screen_coords(corners, ids)
                    if self.screen_transform is not None:
                        self.debug_info = "Screen mapping updated"
                        
                        # Draw the ArUco markers
                        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                        
                        # Draw the screen boundaries
                        try:
                            frame_points = cv2.perspectiveTransform(
                                np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float32), 
                                np.linalg.inv(self.screen_transform)
                            )
                            cv2.polylines(frame, [frame_points.astype(int)], True, (0, 255, 0), 2)
                        except Exception as e:
                            print(f"Error drawing screen boundaries: {e}")

                # Get gaze point and map to screen
                gaze_point = self.shared_gaze_data.get()
                if gaze_point:
                    # Draw the original gaze point
                    gaze_x, gaze_y = tuple(map(int, gaze_point))
                    if 0 <= gaze_x < frame.shape[1] and 0 <= gaze_y < frame.shape[0]:
                        cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 255, 255), -1)
                    
                    # Map gaze to screen coordinates and move cursor
                    if self.screen_transform is not None:
                        screen_point = self.map_gaze_to_screen(gaze_point)
                        if screen_point:
                            self.debug_info = f"Gaze at: {gaze_point}, Screen: {screen_point}"
                            pyautogui.moveTo(screen_point[0], screen_point[1])
                        else:
                            self.debug_info = f"Gaze point out of screen bounds: {gaze_point}"
                    else:
                        self.debug_info = "Screen mapping not available"
                else:
                    self.debug_info = "No gaze point available"

            cv2.putText(frame, self.debug_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(self.preview_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                break

            frame_count += 1

        cam.release()
        cv2.destroyWindow(self.preview_name)

def load_linear_regression_model():
    try:
        lr_model = joblib.load('linearregressionmodelz.joblib')
        print("Linear Regression model loaded successfully.")
        return lr_model
    except Exception as e:
        print(f"Failed to load Linear Regression model: {e}")
        print("Falling back to default gaze projection method.")
        return None

def main(args):
    shared_gaze_data = SharedGazeData()

    camera_matrix = np.array([[343.34511283, 0.0, 327.80111243],
                              [0.0, 342.79698299, 231.06509007],
                              [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([0, 0, 0, -0.001])

    # Load the Linear Regression model
    lr_model = load_linear_regression_model()

    eye_cam_thread = CamThread("Eye Camera", args.eye_cam, args.eye_res, 
                               is_eye_cam=True, focal_length=args.focal_length, 
                               shared_gaze_data=shared_gaze_data,
                               lr_model=lr_model)
    front_cam_thread = CamThread("Front Camera", args.front_cam, args.front_res, 
                                 shared_gaze_data=shared_gaze_data,
                                 camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

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
    parser = argparse.ArgumentParser(description="Gaze-controlled cursor system")
    parser.add_argument("--eye_cam", type=int, default=1, help="Eye camera index")
    parser.add_argument("--front_cam", type=int, default=2, help="Front camera index")
    parser.add_argument("--eye_res", nargs=2, type=int, default=[320, 240], help="Eye camera resolution")
    parser.add_argument("--front_res", nargs=2, type=int, default=[640, 480], help="Front camera resolution")
    parser.add_argument("--focal_length", type=float, default=84, help="Focal length of the eye camera")
    args = parser.parse_args()
    
    main(args)