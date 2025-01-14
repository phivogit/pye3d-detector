import cv2
import threading
import argparse
import numpy as np
from queue import Queue
from pupil_detectors.detector_2d import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
import joblib
import queue


def load_linear_regression_model():
    try:
        lr_model = joblib.load('linearregressionmodel03.joblib')
        print("Linear Regression model loaded successfully.")
        return lr_model
    except Exception as e:
        print(f"Failed to load Linear Regression model: {e}")
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
        self.lr_model = load_linear_regression_model()

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
        if self.lr_model is None:
            return None
        prediction = self.lr_model.predict([gaze_normal])[0]
        return tuple(map(int, prediction))

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

            try:
                gaze_point = self.gaze_queue.get_nowait()
                if 0 <= gaze_point[0] < frame.shape[1] and 0 <= gaze_point[1] < frame.shape[0]:
                    cv2.circle(frame, gaze_point, 15, (0, 0, 255), 3)
            except queue.Empty:
                pass

            cv2.imshow("Front Camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                break

        cam.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False

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
    parser = argparse.ArgumentParser(description="Dual camera eye tracking system")
    parser.add_argument("--eye_cam", type=int, default=1, help="Eye camera index")
    parser.add_argument("--front_cam", type=int, default=2, help="Front camera index")
    parser.add_argument("--eye_res", nargs=2, type=int, default=[320, 240], help="Eye camera resolution")
    parser.add_argument("--front_res", nargs=2, type=int, default=[640, 480], help="Front camera resolution")
    parser.add_argument("--focal_length", type=float, default=69, help="Focal length of the eye camera")
    args = parser.parse_args()
    
    main(args)