import cv2
import threading
import argparse
import numpy as np
import pyautogui
import time
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

class CamThread(threading.Thread):
    def __init__(self, preview_name, stream_url, resolution, is_eye_cam=False, focal_length=None, shared_gaze_data=None, camera_matrix=None, dist_coeffs=None):
        threading.Thread.__init__(self)
        self.preview_name = preview_name
        self.stream_url = stream_url
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
            
            # Initialize the deep learning model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = GazePredictionModel().to(self.device)
            try:
                self.model.load_state_dict(torch.load('best_gaze_model.pth', map_location=self.device))
                print("Deep learning model loaded successfully.")
                self.model.eval()
            except Exception as e:
                print(f"Failed to load deep learning model: {e}")
                self.model = None

    def predict_gaze_point(self, gaze_normal, sphere_center):
        if self.model is None:
            return None
            
        try:
            with torch.no_grad():
                # Normalize inputs
                sphere_center = np.array(sphere_center) / 20.0
                
                # Concatenate and convert to tensor
                features = np.concatenate([gaze_normal, sphere_center])
                input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                
                # Get prediction
                prediction = self.model(input_tensor)
                # Denormalize the output
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
        cam = cv2.VideoCapture(self.stream_url)
        if not cam.isOpened():
            print(f"Error: Could not open stream {self.stream_url}")
            return

        fps = cam.get(cv2.CAP_PROP_FPS) or 60  # Fallback to 60 FPS if server doesn't provide it
        frame_count = 0

        while self.running:
            ret, frame = cam.read()
            if not ret:
                print(f"Failed to grab frame from {self.preview_name}")
                break

            if self.is_eye_cam:
                result_3d = self.process_eye_frame(frame, frame_count, fps)
                # Commented out to reduce lag
                # frame = self.visualize_eye_result(frame, result_3d)
            else:
                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

                gaze_point = self.shared_gaze_data.get()
                if gaze_point:
                    if 0 <= gaze_point[0] < frame.shape[1] and 0 <= gaze_point[1] < frame.shape[0]:
                        cv2.circle(frame, gaze_point, 15, (0, 0, 255), -1)
                        self.debug_info = f"Drawing gaze at: {gaze_point}"
                    else:
                        self.debug_info = f"Gaze point out of bounds: {gaze_point}"
                else:
                    self.debug_info = "No gaze point available"

            cv2.putText(frame, self.debug_info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if not self.is_eye_cam:
                cv2.imshow(self.preview_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                break

            frame_count += 1

        cam.release()
        if not self.is_eye_cam:
            cv2.destroyWindow(self.preview_name)

class GazeControlThread(threading.Thread):
    def __init__(self, shared_gaze_data, disable_failsafe=False):
        threading.Thread.__init__(self)
        self.shared_gaze_data = shared_gaze_data
        self.running = True
        self.last_press_time = 0
        self.a_pressed = False
        self.d_pressed = False
        if disable_failsafe:
            pyautogui.FAILSAFE = False

    def run(self):
        while self.running:
            try:
                gaze_point = self.shared_gaze_data.get()
                if gaze_point:
                    x = gaze_point[0]
                    current_time = time.time()

                    if x < 270:  # Look left - press 'a'
                        if not self.a_pressed:
                            pyautogui.keyDown('a')
                            self.a_pressed = True
                        if self.d_pressed:
                            pyautogui.keyUp('d')
                            self.d_pressed = False
                    elif x > 370:  # Look right - press 'd'
                        if not self.d_pressed:
                            pyautogui.keyDown('d')
                            self.d_pressed = True
                        if self.a_pressed:
                            pyautogui.keyUp('a')
                            self.a_pressed = False
                    else:  # Look center - release both keys
                        if self.a_pressed:
                            pyautogui.keyUp('a')
                            self.a_pressed = False
                        if self.d_pressed:
                            pyautogui.keyUp('d')
                            self.d_pressed = False

            except pyautogui.FailSafeException:
                print("PyAutoGUI fail-safe triggered. Pausing gaze control for 5 seconds.")
                time.sleep(5)
            except Exception as e:
                print(f"An error occurred in gaze control: {e}")

            time.sleep(0.1)  # Small delay to prevent excessive CPU usage

    def stop(self):
        self.running = False
        # Ensure keys are released when stopping
        if self.a_pressed:
            pyautogui.keyUp('a')
        if self.d_pressed:
            pyautogui.keyUp('d')

def main(args):
    shared_gaze_data = SharedGazeData()

    camera_matrix = np.array([[343.34511283, 0.0, 327.80111243],
                             [0.0, 342.79698299, 231.06509007],
                             [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([0, 0, 0, -0.001, -0.0])

    eye_cam_thread = CamThread("Eye Camera", args.eye_stream, args.eye_res, 
                              is_eye_cam=True, focal_length=args.focal_length, 
                              shared_gaze_data=shared_gaze_data)
                              
    front_cam_thread = CamThread("Front Camera", args.front_stream, args.front_res, 
                                shared_gaze_data=shared_gaze_data,
                                camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
                                
    gaze_control_thread = GazeControlThread(shared_gaze_data, disable_failsafe=args.disable_failsafe)

    eye_cam_thread.start()
    front_cam_thread.start()
    gaze_control_thread.start()

    try:
        eye_cam_thread.join()
        front_cam_thread.join()
        gaze_control_thread.join()
    except KeyboardInterrupt:
        print("Stopping threads...")
        eye_cam_thread.stop()
        front_cam_thread.stop()
        gaze_control_thread.stop()
        eye_cam_thread.join()
        front_cam_thread.join()
        gaze_control_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual camera eye tracking system")
    parser.add_argument("--eye_stream", type=str, default="http://192.168.203.53:8081/?action=stream", 
                       help="Eye camera stream URL")
    parser.add_argument("--front_stream", type=str, default="http://192.168.203.53:8080/?action=stream", 
                       help="Front camera stream URL")
    parser.add_argument("--eye_res", nargs=2, type=int, default=[320, 240], 
                       help="Eye camera resolution")
    parser.add_argument("--front_res", nargs=2, type=int, default=[640, 480], 
                       help="Front camera resolution")
    parser.add_argument("--focal_length", type=float, default=84, 
                       help="Focal length of the eye camera")
    parser.add_argument("--disable_failsafe", action="store_true", 
                       help="Disable PyAutoGUI fail-safe")
    args = parser.parse_args()
    
    main(args)