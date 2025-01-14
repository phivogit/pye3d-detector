import cv2
import numpy as np
import json
import os
from pupil_detectors.detector_2d import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
#them vao data
def process_eye_frame(frame, detector_2d, detector_3d, frame_number, fps):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result_2d = detector_2d.detect(grayscale)
    result_2d["timestamp"] = frame_number / fps
    return detector_3d.update_and_detect(result_2d, grayscale)

def process_videos():
    eye_video_path = r'C:\Users\hungn\gaudau\pye3d-detector\eye_output.avi'
    front_video_path = r'C:\Users\hungn\gaudau\pye3d-detector\front_output.avi'
    eye_video = cv2.VideoCapture(eye_video_path)
    front_video = cv2.VideoCapture(front_video_path)

    if not eye_video.isOpened() or not front_video.isOpened():
        print("Failed to open one or both video files")
        return

    # Eye tracking setup
    detector_2d = Detector2D()
    camera = CameraModel(focal_length=84, resolution=(320, 240))
    detector_3d = Detector3D(camera=camera, long_term_mode=DetectorMode.blocking)

    # ArUco setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters = cv2.aruco.DetectorParameters()

    fps = eye_video.get(cv2.CAP_PROP_FPS)
    total_frames = int(eye_video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    output_data = []

    while frame_count < total_frames:
        ret_eye, eye_frame = eye_video.read()
        ret_front, front_frame = front_video.read()

        if not ret_eye or not ret_front:
            break

        frame_count += 1

        # Process frames even before frame 100, but don't record data
        result_3d = process_eye_frame(eye_frame, detector_2d, detector_3d, frame_count, fps)

        # Only start recording data from frame 100 onwards
        if frame_count >= 300:
            # Process front camera frame for ArUco detection
            gray = cv2.cvtColor(front_frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if 'circle_3d' in result_3d and 'normal' in result_3d['circle_3d'] and ids is not None and len(ids) > 0:
                gaze_vector = result_3d['circle_3d']['normal']
                marker_center = np.mean(corners[0][0], axis=0)
                confidence = result_3d['confidence']
                sphere_center = result_3d['sphere']['center']

                if confidence > 0.94:
                    output_data.append({
                        'frame': frame_count,
                        'confidence': confidence,
                        'gaze_direction': list(gaze_vector),
                        'marker_position': marker_center.tolist(),
                        'sphere_center': list(sphere_center)
                    })

    eye_video.release()
    front_video.release()

    # Load existing data if file exists
    filename = 'eye_tracking_datadeep.json'
    existing_data = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            existing_data = json.load(f)

    # Append new data to existing data
    combined_data = existing_data + output_data

    # Save the combined data as JSON
    with open(filename, 'w') as f:
        json.dump(combined_data, f, indent=2)

    print(f"Data appended to {filename}")

if __name__ == "__main__":
    process_videos()