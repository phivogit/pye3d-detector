import cv2
import threading
import argparse
import numpy as np
import os
from pupil_detectors.detector_2d import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
import joblib
import time

class SharedGazeData:
    def __init__(self):
        self.gaze_point = None
        self.pupil_radius = None
        self.lock = threading.Lock()

    def update(self, gaze_point, pupil_radius=None):
        with self.lock:
            self.gaze_point = gaze_point
            self.pupil_radius = pupil_radius

    def get(self):
        with self.lock:
            return self.gaze_point, self.pupil_radius

class VideoThread(threading.Thread):
    def __init__(self, preview_name, video_path, resolution, is_eye_cam=False, focal_length=None, shared_gaze_data=None, camera_matrix=None, dist_coeffs=None, lr_model=None):
        threading.Thread.__init__(self)
        self.preview_name = preview_name
        self.video_path = video_path
        self.resolution = resolution
        self.is_eye_cam = is_eye_cam
        self.shared_gaze_data = shared_gaze_data
        self.running = True
        self.debug_info = ""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.lr_model = lr_model
        self.paused = False
        self.start_event = threading.Event()
        self.last_frame_time = 0
        self.real_time_factor = 1.0  # Default to real-time playback

        if is_eye_cam:
            self.detector_2d = Detector2D()
            self.camera = CameraModel(focal_length=focal_length, resolution=resolution)
            self.detector_3d = Detector3D(camera=self.camera, long_term_mode=DetectorMode.blocking)

    def run(self):
        print(f'Starting {self.preview_name}')
        # Wait for the start signal
        self.start_event.wait()
        self.video_preview()

    def stop(self):
        self.running = False

    def process_eye_frame(self, frame, frame_number, fps):
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result_2d = self.detector_2d.detect(grayscale)
        result_2d["timestamp"] = frame_number / fps
        result_3d = self.detector_3d.update_and_detect(result_2d, grayscale)
        
        pupil_radius = None
        if result_3d['confidence'] > 0.6:
            # Extract pupil radius from 3D model if available
            if 'circle_3d' in result_3d and 'radius' in result_3d['circle_3d']:
                pupil_radius = result_3d['circle_3d']['radius']
                # Only print pupil radius to terminal
                print(f"Pupil radius: {pupil_radius:.2f} mm")
            
            if 'circle_3d' in result_3d and 'normal' in result_3d['circle_3d']:
                gaze_normal = result_3d['circle_3d']['normal']
                gaze_point = self.predict_gaze_point(gaze_normal)
                if gaze_point is not None:
                    self.shared_gaze_data.update(gaze_point, pupil_radius)
        
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

    def video_preview(self):
        if not os.path.exists(self.video_path):
            print(f"Error: Video file {self.video_path} does not exist")
            return
            
        video = cv2.VideoCapture(self.video_path)
        if not video.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return

        # Get the original video frame rate
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default to 30 fps if we can't determine it
            
        # Get original resolution
        orig_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_count = 0
        frame_time = 1.0 / fps  # seconds between frames for real-time playback
        
        cv2.namedWindow(self.preview_name, cv2.WINDOW_NORMAL)
        if not self.is_eye_cam:
            cv2.resizeWindow(self.preview_name, 640, 480)
        else:
            cv2.resizeWindow(self.preview_name, 320, 240)
        
        self.last_frame_time = time.time()
        
        while self.running:
            if not self.paused:
                # Calculate time to wait for real-time playback
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                sleep_time = max(0, frame_time - elapsed)
                
                # Sleep precisely for real-time playback
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                ret, frame = video.read()
                if not ret:
                    print(f"End of video or error reading from {self.preview_name}")
                    break
                
                self.last_frame_time = time.time()
                    
                # Resize frame if needed and only for eye camera
                if self.is_eye_cam and self.resolution != (orig_width, orig_height):
                    frame = cv2.resize(frame, self.resolution)

                if self.is_eye_cam:
                    try:
                        result_3d = self.process_eye_frame(frame, frame_count, fps)
                        frame = self.visualize_eye_result(frame, result_3d)
                    except Exception as e:
                        print(f"Error processing eye frame: {e}")
                else:
                    if self.camera_matrix is not None and self.dist_coeffs is not None:
                        frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

                    gaze_data = self.shared_gaze_data.get()
                    gaze_point = gaze_data[0]
                    
                    if gaze_point:
                        gaze_point = tuple(map(int, gaze_point))
                        if 0 <= gaze_point[0] < frame.shape[1] and 0 <= gaze_point[1] < frame.shape[0]:
                            cv2.circle(frame, gaze_point, 15, (0, 0, 255), -1)

                frame_count += 1
                
            try:
                cv2.imshow(self.preview_name, frame)
            except Exception as e:
                print(f"Error displaying frame in {self.preview_name}: {e}")
            
            # Handle keyboard input with minimal processing time
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Press 'Esc' to exit
                self.running = False
                break
            elif key == 32:  # Press 'Space' to pause/resume
                self.paused = not self.paused
                print(f"{self.preview_name}: {'Paused' if self.paused else 'Resumed'}")

        video.release()
        cv2.destroyWindow(self.preview_name)
        print(f"Analysis complete for {self.preview_name}.")

def load_linear_regression_model():
    try:
        lr_model = joblib.load('linearregressionmodeldeepX.joblib')
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
    if lr_model is None:
        print("Error: Linear Regression model is required for gaze projection")
        return

    # Create threads but don't start them yet
    eye_video_thread = VideoThread("Eye Video", args.eye_video, tuple(args.eye_res), 
                                  is_eye_cam=True, focal_length=args.focal_length, 
                                  shared_gaze_data=shared_gaze_data,
                                  lr_model=lr_model)
    front_video_thread = VideoThread("Front Video", args.front_video, tuple(args.front_res), 
                                    shared_gaze_data=shared_gaze_data,
                                    camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    # Start threads
    eye_video_thread.start()
    front_video_thread.start()
    
    # Ready to start videos
    print("Starting both videos simultaneously in 1 second...")
    time.sleep(1)
    
    # Signal both threads to start playing videos simultaneously
    eye_video_thread.start_event.set()
    front_video_thread.start_event.set()
    print("Both videos started!")

    try:
        eye_video_thread.join()
        front_video_thread.join()
    except KeyboardInterrupt:
        print("Stopping threads...")
        eye_video_thread.stop()
        front_video_thread.stop()
        eye_video_thread.join()
        front_video_thread.join()
    except Exception as e:
        print(f"Error in main thread: {e}")
        eye_video_thread.stop()
        front_video_thread.stop()

    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual video eye tracking system")
    parser.add_argument("--eye_video", type=str, required=True, help="Path to eye video file")
    parser.add_argument("--front_video", type=str, required=True, help="Path to front video file")
    parser.add_argument("--eye_res", nargs=2, type=int, default=[320, 240], help="Eye video resolution")
    parser.add_argument("--front_res", nargs=2, type=int, default=[640, 480], help="Front video resolution")
    parser.add_argument("--focal_length", type=float, default=84, help="Focal length of the eye camera")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save output videos")
    args = parser.parse_args()
    
    main(args)
    
    #python examples/RYR.py --eye_video C:\Users\hungn\gaudau\pye3d-detector\camera2_320x240_20250307_193414.mp4 --front_video C:\Users\hungn\gaudau\pye3d-detector\camera1_640x480_20250307_193414.mp4
    #python examples/RYR.py --eye_video C:\Users\hungn\Downloads\camera2_320x240_20250310_100418.mp4 --front_video C:\Users\hungn\Downloads\camera1_640x480_20250310_100418.mp4