import cv2
import numpy as np
import pyautogui
import threading
import time
from pynput.mouse import Controller
from pupil_detectors.detector_2d import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
import torch

class GazePredictionModel(torch.nn.Module):
    def __init__(self, hidden_size=128):
        super(GazePredictionModel, self).__init__()
        
        self.input_size = 6  # gaze_direction (3) + sphere_center (3)
        self.output_size = 2  # marker_position (2)
        
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size, hidden_size//2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_size//2),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size//2, hidden_size//4),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_size//4),
            torch.nn.Linear(hidden_size//4, self.output_size)
        )
        
    def forward(self, x):
        return self.network(x)

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
                 shared_gaze_data=None, camera_matrix=None, dist_coeffs=None):
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
        
        if is_eye_cam:
            self.detector_2d = Detector2D()
            self.camera = CameraModel(focal_length=focal_length, resolution=resolution)
            self.detector_3d = Detector3D(camera=self.camera, long_term_mode=DetectorMode.blocking)
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = GazePredictionModel().to(self.device)
            self.model.load_state_dict(torch.load('best_gaze_model2.pth', map_location=self.device))
            self.model.eval()

    def predict_gaze_point(self, gaze_normal, sphere_center):
        try:
            with torch.no_grad():
                sphere_center = np.array(sphere_center) / 20.0
                features = np.concatenate([gaze_normal, sphere_center])
                input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                prediction = self.model(input_tensor)
                prediction = prediction.cpu().squeeze().numpy() * 500.0
                return tuple(map(int, prediction))
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def process_eye_frame(self, frame, frame_number, fps):
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result_2d = self.detector_2d.detect(grayscale)
        result_2d["timestamp"] = frame_number / fps
        result_3d = self.detector_3d.update_and_detect(result_2d, grayscale)
        
        if result_3d['confidence'] > 0.6 and 'circle_3d' in result_3d:
            gaze_normal = result_3d['circle_3d']['normal']
            sphere_center = result_3d['sphere']['center']
            
            gaze_point = self.predict_gaze_point(gaze_normal, sphere_center)
            if gaze_point is not None:
                self.shared_gaze_data.update(gaze_point)
                self.debug_info = f"Predicted gaze point: {gaze_point}"
            else:
                self.debug_info = "Invalid gaze prediction"
        else:
            self.debug_info = "Low confidence or missing gaze data"
        
        return result_3d

    def run(self):
        self.cam_preview()

    def stop(self):
        self.running = False

    def cam_preview(self):
        cap = cv2.VideoCapture(self.cam_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.cam_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        if not self.is_eye_cam:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        frame_count = 0
        fps = 60

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            if self.is_eye_cam:
                result_3d = self.process_eye_frame(frame, frame_count, fps)
                if 'ellipse' in result_3d:
                    ellipse = result_3d["ellipse"]
                    cv2.ellipse(frame, 
                              tuple(int(v) for v in ellipse["center"]),
                              tuple(int(v/2) for v in ellipse["axes"]),
                              ellipse["angle"], 0, 360, (0, 255, 0), 2)
            else:
                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
                
                gaze_point = self.shared_gaze_data.get()
                if gaze_point:
                    cv2.circle(frame, gaze_point, 15, (0, 0, 255), -1)

            cv2.putText(frame, self.debug_info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(self.preview_name, frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_count += 1

        cap.release()
        cv2.destroyWindow(self.preview_name)

class DisplayDetector:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.kernel = np.ones((5,5), np.uint8)
        self.min_area = 640 * 480 * 0.1
        self.display_corners = None

    def detect_corners(self, frame):
        frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        _, white_regions = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 
                                      200, 255, cv2.THRESH_BINARY)
        white_regions = cv2.morphologyEx(white_regions, cv2.MORPH_CLOSE, self.kernel)
        contours, _ = cv2.findContours(white_regions, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > self.min_area:
                approx = cv2.approxPolyDP(largest_contour, 
                                        0.02 * cv2.arcLength(largest_contour, True), 
                                        True)
                if len(approx) == 4:
                    pts = np.float32(approx[:, 0])
                    rect = np.zeros((4, 2), dtype="float32")
                    s = pts.sum(axis=1)
                    rect[0] = pts[np.argmin(s)]
                    rect[2] = pts[np.argmax(s)]
                    diff = np.diff(pts, axis=1)
                    rect[1] = pts[np.argmin(diff)]
                    rect[3] = pts[np.argmax(diff)]
                    self.display_corners = rect
                    return rect
        return None

    def get_display_corners(self):
        return self.display_corners

class GazeMouseController:
    def __init__(self, display_resolution=(1920, 1080), camera_resolution=(640, 480)):
        self.mouse = Controller()
        self.display_resolution = display_resolution
        self.camera_resolution = camera_resolution
        self.shared_gaze_data = SharedGazeData()
        self.running = True
        
        self.camera_matrix = np.array([
            [343.34511283, 0.0, 327.80111243],
            [0.0, 342.79698299, 231.06509007],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.array([0, 0, 0, -0.001, -0.0])
        
        self.display_detector = DisplayDetector(self.camera_matrix, self.dist_coeffs)
        
        self.smoothing_window = []
        self.window_size = 5
        self.min_movement = 5
        
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.1

    def smooth_coordinates(self, x, y):
        self.smoothing_window.append((x, y))
        if len(self.smoothing_window) > self.window_size:
            self.smoothing_window.pop(0)
        if len(self.smoothing_window) < self.window_size:
            return x, y
        smooth_x = sum(p[0] for p in self.smoothing_window) / len(self.smoothing_window)
        smooth_y = sum(p[1] for p in self.smoothing_window) / len(self.smoothing_window)
        return int(smooth_x), int(smooth_y)

    def map_gaze_to_screen(self, gaze_point):
        """Map camera coordinates to screen coordinates using normalized perspective transform"""
        if not gaze_point:
            return None
        
        display_corners = self.display_detector.get_display_corners()
        if display_corners is None:
            # Fall back to simple scaling if no display corners detected
            screen_x = int(gaze_point[0] * (self.display_resolution[0] / self.camera_resolution[0]))
            screen_y = int(gaze_point[1] * (self.display_resolution[1] / self.camera_resolution[1]))
        else:
            # Normalize the destination points to [0, 1] range
            dst_points = np.array([
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1]
            ], dtype=np.float32)
            
            # Get perspective transform matrix using normalized coordinates
            matrix = cv2.getPerspectiveTransform(display_corners, dst_points)
            
            # Transform the gaze point using the normalized matrix
            point = np.array([[[gaze_point[0], gaze_point[1]]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, matrix)
            normalized_x, normalized_y = transformed[0][0]
            
            # Convert normalized coordinates back to screen coordinates
            screen_x = normalized_x * self.display_resolution[0]
            screen_y = normalized_y * self.display_resolution[1]
        
        # Ensure coordinates are within screen bounds
        screen_x = max(0, min(screen_x, self.display_resolution[0]))
        screen_y = max(0, min(screen_y, self.display_resolution[1]))
        
        return int(screen_x), int(screen_y)

    def update_mouse_position(self):
        last_position = None
        while self.running:
            gaze_point = self.shared_gaze_data.get()
            if gaze_point:
                screen_coords = self.map_gaze_to_screen(gaze_point)
                if screen_coords:
                    smooth_x, smooth_y = self.smooth_coordinates(*screen_coords)
                    if last_position:
                        dx = abs(smooth_x - last_position[0])
                        dy = abs(smooth_y - last_position[1])
                        if dx > self.min_movement or dy > self.min_movement:
                            pyautogui.moveTo(smooth_x, smooth_y)
                            last_position = (smooth_x, smooth_y)
                    else:
                        pyautogui.moveTo(smooth_x, smooth_y)
                        last_position = (smooth_x, smooth_y)
            time.sleep(0.016)

    def start(self, eye_cam_id=1, front_cam_id=2):
        eye_cam_thread = CamThread(
            "Eye Camera", 
            eye_cam_id, 
            (320, 240), 
            is_eye_cam=True,
            focal_length=84,
            shared_gaze_data=self.shared_gaze_data
        )
        
        front_cam_thread = CamThread(
            "Front Camera",
            front_cam_id,
            self.camera_resolution,
            shared_gaze_data=self.shared_gaze_data,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs
        )
        
        mouse_thread = threading.Thread(target=self.update_mouse_position)
        
        try:
            eye_cam_thread.start()
            front_cam_thread.start()
            mouse_thread.start()
            
            eye_cam_thread.join()
            front_cam_thread.join()
            mouse_thread.join()
            
        except KeyboardInterrupt:
            print("Stopping gaze mouse control...")
            self.running = False
            eye_cam_thread.stop()
            front_cam_thread.stop()
            
            eye_cam_thread.join()
            front_cam_thread.join()
            mouse_thread.join()

def main():
    controller = GazeMouseController(
        display_resolution=(1920, 1080),
        camera_resolution=(640, 480)
    )
    
    try:
        controller.start()
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()