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

        # ArUco dictionary
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        self.aruco_params = aruco.DetectorParameters()

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
            
            try:
                gaze_point = self.gaze_queue.get_nowait()
            except queue.Empty:
                pass

            if gaze_point is not None:
                if 0 <= gaze_point[0] < frame.shape[1] and 0 <= gaze_point[1] < frame.shape[0]:
                    cv2.circle(frame, gaze_point, 15, (0, 0, 255), -1)
                
                # Check if gaze point is on any detected marker
                if ids is not None:
                    for i, corner in enumerate(corners):
                        if self.point_in_polygon(gaze_point, corner[0]):
                            marker_id = ids[i][0]
                            if marker_id == 2:
                                print("Looking at id2")
                            elif marker_id == 5:
                                print("Looking at id5")

            cv2.imshow("Front Camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                break

        cam.release()
        cv2.destroyAllWindows()
    def stop(self):
        self.running = False

    def point_in_polygon(self, point, polygon):
        # Calculate center of the marker
        center_x = sum(p[0] for p in polygon) / len(polygon)
        center_y = sum(p[1] for p in polygon) / len(polygon)
        
        # Create expanded polygon (scale factor > 1.0 increases the region)
        scale_factor = 1.5  # Adjust this value to make region bigger or smaller
        expanded_polygon = []
        for px, py in polygon:
            # Calculate vector from center to point
            dx = px - center_x
            dy = py - center_y
            # Scale the vector and add back to center
            expanded_x = center_x + (dx * scale_factor)
            expanded_y = center_y + (dy * scale_factor)
            expanded_polygon.append((expanded_x, expanded_y))
        
        # Original point-in-polygon test with expanded polygon
        x, y = point
        n = len(expanded_polygon)
        inside = False
        p1x, p1y = expanded_polygon[0]
        for i in range(n + 1):
            p2x, p2y = expanded_polygon[i % n]
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
    parser = argparse.ArgumentParser(description="Dual camera eye tracking system")
    parser.add_argument("--eye_cam", type=int, default=2, help="Eye camera index")
    parser.add_argument("--front_cam", type=int, default=1, help="Front camera index")
    parser.add_argument("--eye_res", nargs=2, type=int, default=[320, 240], help="Eye camera resolution")
    parser.add_argument("--front_res", nargs=2, type=int, default=[640, 480], help="Front camera resolution")
    parser.add_argument("--focal_length", type=float, default=184.7, help="Focal length of the eye camera")
    args = parser.parse_args()
    
    main(args)