import cv2
import threading
import argparse
import numpy as np
import torch
from pupil_detectors.detector_2d import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode

class GazePredictionModel(torch.nn.Module):
    def __init__(self, hidden_size=256):
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

class TestCamThread(threading.Thread):
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
        self.marker_size = 0.046  # 4.6 cm
        
        if is_eye_cam:
            self.detector_2d = Detector2D()
            self.camera = CameraModel(focal_length=focal_length, resolution=resolution)
            self.detector_3d = Detector3D(camera=self.camera, long_term_mode=DetectorMode.blocking)
            
            # Initialize the model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = GazePredictionModel().to(self.device)
            self.model.load_state_dict(torch.load('best_gaze_model.pth', 
                                                map_location=self.device))
            self.model.eval()
        else:
            # Initialize ArUco detector
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
            self.aruco_params = cv2.aruco.DetectorParameters()

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

    def detect_aruco_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        marker_centers = {}
        if ids is not None:
            for i in range(len(ids)):
                if ids[i][0] == 6:  # Only interested in marker ID 6
                    marker_corners = corners[i][0]
                    center = np.mean(marker_corners, axis=0)
                    marker_centers[ids[i][0]] = tuple(map(int, center))
        
        return marker_centers

    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def pixel_to_cm(self, pixel_distance):
        return (pixel_distance * self.marker_size * 100) / self.camera_matrix[0][0]

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
                self.debug_info = f"Predicted: {gaze_point}"
            else:
                self.debug_info = "Invalid prediction"
        else:
            self.debug_info = "Low confidence or missing data"
        
        return result_3d

    def visualize_eye_result(self, frame, result_3d):
        if 'ellipse' in result_3d:
            ellipse = result_3d["ellipse"]
            cv2.ellipse(frame, 
                      tuple(int(v) for v in ellipse["center"]),
                      tuple(int(v / 2) for v in ellipse["axes"]),
                      ellipse["angle"], 0, 360, (0, 255, 0), 2)
        return frame

    def run(self):
        print(f'Starting {self.preview_name}')
        self.cam_preview()

    def stop(self):
        self.running = False

    def cam_preview(self):
        cam = cv2.VideoCapture(self.cam_id)
        if not cam.isOpened():
            print(f"Error: Could not open camera {self.cam_id}")
            return

        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        if not self.is_eye_cam:
            cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        fps = 60
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

                marker_centers = self.detect_aruco_markers(frame)
                gaze_point = self.shared_gaze_data.get()

                # Draw ArUco marker center (blue) and gaze point (red)
                if 6 in marker_centers:
                    cv2.circle(frame, marker_centers[6], 5, (255, 0, 0), -1)
                if gaze_point:
                    cv2.circle(frame, gaze_point, 5, (0, 0, 255), -1)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    if 6 in marker_centers and gaze_point:
                        distance_pixels = self.calculate_distance(marker_centers[6], gaze_point)
                        distance_cm = self.pixel_to_cm(distance_pixels)
                        print(f"Distance between ArUco ID 6 and gaze point: {distance_pixels:.2f} pixels, {distance_cm:.2f} cm")
                    else:
                        print("ArUco marker ID 6 or gaze point not detected.")
                elif key == 27:  # Press 'Esc' to exit
                    break

            cv2.putText(frame, self.debug_info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(self.preview_name, frame)
            
            frame_count += 1

        cam.release()
        cv2.destroyWindow(self.preview_name)

def main(args):
    shared_gaze_data = SharedGazeData()

    camera_matrix = np.array([[343.34511283, 0.0, 327.80111243],
                             [0.0, 342.79698299, 231.06509007],
                             [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([0, 0, 0, -0.001, -0.0])

    eye_cam_thread = TestCamThread("Eye Camera", args.eye_cam, args.eye_res, 
                                  is_eye_cam=True, focal_length=args.focal_length, 
                                  shared_gaze_data=shared_gaze_data)
    
    front_cam_thread = TestCamThread("Front Camera", args.front_cam, args.front_res, 
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
    parser = argparse.ArgumentParser(description="Test version of dual camera eye tracking system")
    parser.add_argument("--eye_cam", type=int, default=1, help="Eye camera index")
    parser.add_argument("--front_cam", type=int, default=2, help="Front camera index")
    parser.add_argument("--eye_res", nargs=2, type=int, default=[320, 240], help="Eye camera resolution")
    parser.add_argument("--front_res", nargs=2, type=int, default=[640, 480], help="Front camera resolution")
    parser.add_argument("--focal_length", type=float, default=84, help="Focal length of the eye camera")
    args = parser.parse_args()
    
    main(args)