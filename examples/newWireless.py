import cv2
import threading
import numpy as np
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
    def __init__(self, preview_name, stream_url, is_eye_cam=False, focal_length=None, resolution=(640, 480), shared_gaze_data=None, camera_matrix=None, dist_coeffs=None, lr_model=None):
        threading.Thread.__init__(self)
        self.preview_name = preview_name
        self.stream_url = stream_url
        self.is_eye_cam = is_eye_cam
        self.shared_gaze_data = shared_gaze_data
        self.running = True
        self.debug_info = ""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.lr_model = lr_model
        self.resolution = resolution

        if is_eye_cam:
            self.detector_2d = Detector2D()
            self.camera = CameraModel(focal_length=focal_length, resolution=resolution)
            self.detector_3d = Detector3D(camera=self.camera, long_term_mode=DetectorMode.blocking)

    def run(self):
        print(f"Starting {self.preview_name}")
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

    def visualize_eye_result(self, frame, result_3d):
        if 'ellipse' in result_3d:
            ellipse = result_3d["ellipse"]
            cv2.ellipse(frame, 
                        tuple(int(v) for v in ellipse["center"]),
                        tuple(int(v / 2) for v in ellipse["axes"]),
                        ellipse["angle"], 0, 360, (0, 255, 0), 2)
        return frame

    def cam_preview(self):
        cam = cv2.VideoCapture(self.stream_url)
        if not cam.isOpened():
            print(f"Error: Could not open stream {self.stream_url}")
            return

        frame_count = 0
        fps = cam.get(cv2.CAP_PROP_FPS) or 30  # Fallback to 30 FPS if server doesn't provide it

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

                gaze_point = self.shared_gaze_data.get()
                if gaze_point:
                    gaze_point = tuple(map(int, gaze_point))
                    if 0 <= gaze_point[0] < frame.shape[1] and 0 <= gaze_point[1] < frame.shape[0]:
                        cv2.circle(frame, gaze_point, 15, (0, 0, 255), -1)
                        self.debug_info = f"Drawing gaze at: {gaze_point}"
                    else:
                        self.debug_info = f"Gaze point out of bounds: {gaze_point}"
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
        lr_model = joblib.load('linearregressionmodeldeep2.joblib')
        print("Linear Regression model loaded successfully.")
        return lr_model
    except Exception as e:
        print(f"Failed to load Linear Regression model: {e}")
        print("Falling back to default gaze projection method.")
        return None

def main():
    shared_gaze_data = SharedGazeData()

    # Hardcoded HTTP stream URLs
    eye_stream_url = "http://192.168.89.68:8081/?action=stream"
    front_stream_url = "http://192.168.89.68:8080/?action=stream"

    # Resolutions
    eye_resolution = (320, 240)
    front_resolution = (640, 480)

    camera_matrix = np.array([[343.34511283, 0.0, 327.80111243],
                              [0.0, 342.79698299, 231.06509007],
                              [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([0, 0, 0, -0.001, -0.0])

    # Load the Linear Regression model
    lr_model = load_linear_regression_model()

    # Create threads for the eye and front cameras
    eye_cam_thread = CamThread("Eye Camera", eye_stream_url, 
                               is_eye_cam=True, focal_length=84, resolution=eye_resolution,
                               shared_gaze_data=shared_gaze_data,
                               lr_model=lr_model)
    front_cam_thread = CamThread("Front Camera", front_stream_url, 
                                 resolution=front_resolution,
                                 shared_gaze_data=shared_gaze_data,
                                 camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    # Start the threads
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
    main()
