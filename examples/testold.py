import cv2
import threading
import argparse
import numpy as np
from pupil_detectors.detector_2d import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode

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

class EyeCamThread(threading.Thread):
    def __init__(self, cam_id, resolution, focal_length, shared_gaze_data):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.resolution = resolution
        self.shared_gaze_data = shared_gaze_data
        self.running = True
        self.detector_2d = Detector2D()
        self.camera = CameraModel(focal_length=focal_length, resolution=resolution)
        self.detector_3d = Detector3D(camera=self.camera, long_term_mode=DetectorMode.blocking)

    def run(self):
        print('Starting Eye Camera Thread')
        self.process_eye_cam()

    def stop(self):
        self.running = False

    def process_eye_frame(self, frame, frame_number, fps):
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result_2d = self.detector_2d.detect(grayscale)
        result_2d["timestamp"] = frame_number / fps
        result_3d = self.detector_3d.update_and_detect(result_2d, grayscale)
        
        if result_3d['confidence'] > 0.6 and 'circle_3d' in result_3d and 'normal' in result_3d['circle_3d']:
            gaze_normal = result_3d['circle_3d']['normal']
            gaze_point = self.calculate_gaze_point(gaze_normal)
            self.shared_gaze_data.update(gaze_point)

    def calculate_gaze_point(self, gaze_normal):
        t = -0.95 / gaze_normal[2]  # Using the optimized plane_z value

        X = float(gaze_normal[0] * t)
        Y = float(gaze_normal[1] * t)

        X_transformed = X + (0.0868 * X**2)  # Using the optimized curve_x
        Y_transformed = Y + (0.0062 * Y**2)  # Using the optimized curve_y

        x_2d = int(330.31 + (-X_transformed * 320.15))  # Using optimized offset_x and scale_x
        y_2d = int(389.91 + (Y_transformed * 315.78)) 
        return (x_2d, y_2d)


    def process_eye_cam(self):
        cam = cv2.VideoCapture(self.cam_id)
        if not cam.isOpened():
            print(f"Error: Could not open eye camera {self.cam_id}")
            return

        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        fps = 60
        frame_count = 0

        while self.running:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame from eye camera")
                break

            self.process_eye_frame(frame, frame_count, fps)
            frame_count += 1

        cam.release()

class FrontCamThread(threading.Thread):
    def __init__(self, cam_id, resolution, shared_gaze_data, camera_matrix, dist_coeffs):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        self.resolution = resolution
        self.shared_gaze_data = shared_gaze_data
        self.running = True
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.marker_size = 0.046  # 4.6 cm

    def run(self):
        print('Starting Front Camera Thread')
        self.process_front_cam()

    def stop(self):
        self.running = False

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

    def process_front_cam(self):
        cam = cv2.VideoCapture(self.cam_id)
        if not cam.isOpened():
            print(f"Error: Could not open front camera {self.cam_id}")
            return

        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        while self.running:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame from front camera")
                break

            if self.camera_matrix is not None and self.dist_coeffs is not None:
                frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

            marker_centers = self.detect_aruco_markers(frame)
            gaze_point = self.shared_gaze_data.get()

            if gaze_point:
                cv2.circle(frame, gaze_point, 5, (0, 0, 255), -1)

            if 6 in marker_centers:
                cv2.circle(frame, marker_centers[6], 5, (255, 0, 0), -1)

            cv2.imshow("Front Camera", frame)

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

        cam.release()
        cv2.destroyAllWindows()

def main(args):
    shared_gaze_data = SharedGazeData()

    camera_matrix = np.array([[343.34511283, 0.0, 327.80111243],
                              [0.0, 342.79698299, 231.06509007],
                              [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([0, 0, 0, -0.001, -0.0])

    eye_cam_thread = EyeCamThread(args.eye_cam, args.eye_res, args.focal_length, 
                                  shared_gaze_data)
    front_cam_thread = FrontCamThread(args.front_cam, args.front_res, 
                                      shared_gaze_data, camera_matrix, dist_coeffs)

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
    parser = argparse.ArgumentParser(description="Dual camera eye tracking system with basic gaze calculation and ArUco marker detection")
    parser.add_argument("--eye_cam", type=int, default=1, help="Eye camera index")
    parser.add_argument("--front_cam", type=int, default=2, help="Front camera index")
    parser.add_argument("--eye_res", nargs=2, type=int, default=[320, 240], help="Eye camera resolution")
    parser.add_argument("--front_res", nargs=2, type=int, default=[640, 480], help="Front camera resolution")
    parser.add_argument("--focal_length", type=float, default=84, help="Focal length of the eye camera")
    args = parser.parse_args()
    
    main(args)