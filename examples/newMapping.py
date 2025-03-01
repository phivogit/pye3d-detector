import cv2
import threading
import argparse
import numpy as np
from pupil_detectors.detector_2d import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode

class FrontCameraIntrinsics:
    """Front camera intrinsics model with specific calibration parameters"""
    def __init__(self, resolution=(1280, 720)):
        self.resolution = resolution
        
        # Use the provided camera matrix
        self.camera_matrix = np.array([[576.0939971760606, 0.0, 655.1495459573105],
                          [0.0, 559.9657414320292, 362.99899317192825],
                          [0.0, 0.0, 1.0]])

        # Use the provided distortion coefficients
        self.dist_coeffs = np.array([0, 0, 0, -0.001])
        
        # Extract focal length from camera matrix
        self.focal_length = self.camera_matrix[0, 0]
    
    def projectPoints(self, object_points, rvec, tvec):
        rvec = np.zeros(3).reshape(1, 1, 3)
        tvec = np.zeros(3).reshape(1, 1, 3)
        
        return cv2.projectPoints(object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)[0]


class GazeMapper:
    def __init__(self, front_camera_resolution=(1280, 720)):
        # Hardcoded eye_camera_to_world_matrix from calibration
        self.eye_camera_to_world_matrix = np.array([
            [-0.42760081742403516, -0.7848729415882221, -0.4484774314287715, -29.731415636833695],
            [-0.5043020103124323,  0.6188632685578088, -0.6022356160380957, 51.91357779601506],
            [0.7502246485774354, -0.03134837145047159, -0.6604394417918019, 99.3595813808341],
            [0.0, 0.0, 0.0, 1.0]])
        self.gaze_distance = 500
        
        # Extract rotation and translation components
        self.rotation_matrix = self.eye_camera_to_world_matrix[:3, :3]
        self.rotation_vector = cv2.Rodrigues(self.rotation_matrix)[0]
        self.translation_vector = self.eye_camera_to_world_matrix[:3, 3]
        
        # Create front camera intrinsics
        self.front_camera_intrinsics = FrontCameraIntrinsics(resolution=front_camera_resolution)
    
    def map_gaze(self, result_3d):
        """Maps 3D pupil detection result to gaze points using calibration data"""
        if 'circle_3d' not in result_3d or 'normal' not in result_3d['circle_3d'] or 'sphere' not in result_3d or 'center' not in result_3d['sphere']:
            return None
        
        # Extract pupil normal and sphere center
        pupil_normal = np.array(result_3d['circle_3d']['normal'])
        sphere_center = np.array(result_3d['sphere']['center'])
        
        # Calculate gaze point in eye camera coordinates
        gaze_point = pupil_normal * self.gaze_distance + sphere_center
        

        # Project 3D gaze point to 2D using front camera intrinsics
        image_point = self.front_camera_intrinsics.projectPoints(
            gaze_point[np.newaxis].reshape(-1, 1, 3),  # Keep the original gaze_point in eye coordinates
            self.rotation_vector,                       # Use the eye rotation vector
            self.translation_vector                     # Use the eye translation vector
        )
        image_point = image_point.reshape(-1, 2)
        
        return (int(image_point[0][0]), int(image_point[0][1]))
    
    def _to_world(self, point):
        """Transform a point from eye camera to world coordinates"""
        p = np.ones(4)
        p[:3] = point[:3]
        return np.dot(self.eye_camera_to_world_matrix, p)[:3]

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
    def __init__(self, preview_name, cam_id, resolution, is_eye_cam=False, focal_length=None, shared_gaze_data=None):
        threading.Thread.__init__(self)
        self.preview_name = preview_name
        self.cam_id = cam_id
        self.resolution = resolution
        self.is_eye_cam = is_eye_cam
        self.shared_gaze_data = shared_gaze_data
        self.running = True
        self.debug_info = ""

        if is_eye_cam:
            self.detector_2d = Detector2D()
            self.camera = CameraModel(focal_length=focal_length, resolution=resolution)
            self.detector_3d = Detector3D(camera=self.camera, long_term_mode=DetectorMode.blocking)
            self.gaze_mapper = GazeMapper(front_camera_resolution=(1280, 720))

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
        
        if result_3d['confidence'] > 0.6:
            gaze_point = self.gaze_mapper.map_gaze(result_3d)
            if gaze_point is not None:
                self.shared_gaze_data.update(gaze_point)
                self.debug_info = f"Mapped gaze point: {gaze_point}"
            else:
                self.debug_info = "Invalid gaze mapping"
        else:
            self.debug_info = "Low confidence"
        
        return result_3d

    def visualize_eye_result(self, frame, result_3d):
        if 'ellipse' in result_3d:
            ellipse = result_3d["ellipse"]
            cv2.ellipse(frame, 
                        tuple(int(v) for v in ellipse["center"]),
                        tuple(int(v / 2) for v in ellipse["axes"]),
                        ellipse["angle"], 0, 360, (0, 255, 0), 2)
        return frame

    def cam_preview(self):
        cam = cv2.VideoCapture(self.cam_id)
        if not cam.isOpened():
            print(f"Error: Could not open camera {self.cam_id}")
            return

        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Disable autofocus for the front camera
        if not self.is_eye_cam:
            cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

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
                gaze_point = self.shared_gaze_data.get()
                if gaze_point:
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

def main(args):
    shared_gaze_data = SharedGazeData()

    eye_cam_thread = CamThread("Eye Camera", args.eye_cam, args.eye_res, 
                              is_eye_cam=True, focal_length=args.focal_length, 
                              shared_gaze_data=shared_gaze_data)
    front_cam_thread = CamThread("Front Camera", args.front_cam, args.front_res, 
                                shared_gaze_data=shared_gaze_data)

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
    parser.add_argument("--front_res", nargs=2, type=int, default=[1280, 720], help="Front camera resolution")
    parser.add_argument("--focal_length", type=float, default=84, help="Focal length of the eye camera")
    args = parser.parse_args()
    
    main(args)