import cv2
import numpy as np
from pupil_apriltags import Detector

# Camera calibration parameters from the previous calibration step
camera_matrix = np.array([[343.34511283, 0.0, 327.80111243],
                          [0.0, 342.79698299, 231.06509007],
                          [0.0, 0.0, 1.0]])
dist_coeffs = np.array([0, 0, 0, -0.001])

# Initialize camera
camera_id = 1
cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print(f"Error: Cannot open camera with ID {camera_id}")
    exit()

# Disable autofocus
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0 = disable autofocus

# Set up AprilTag detector for 36h11 family
detector = Detector(families="tag36h11")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Undistort the frame
    frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Convert to grayscale
    gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    detections = detector.detect(gray)

    # Draw detected markers and axes
    if detections:
        for det in detections:
            # Draw the tag boundary
            corners = np.array(det.corners, dtype=int)
            cv2.polylines(frame_undistorted, [corners], True, (0, 255, 0), 2)
            # Draw the tag ID at the center
            center = det.center.astype(int)
            cv2.putText(frame_undistorted, str(det.tag_id), tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Estimate pose
            tag_size = 0.05  # Size of the tag in meters
            corners_float = det.corners.reshape(1, -1, 2)
            _, rvec, tvec = cv2.solvePnP(
                np.array([[-tag_size/2, -tag_size/2, 0],
                          [tag_size/2, -tag_size/2, 0],
                          [tag_size/2, tag_size/2, 0],
                          [-tag_size/2, tag_size/2, 0]], dtype=np.float32),
                corners_float,
                camera_matrix,
                dist_coeffs
            )
            # Draw coordinate axes
            cv2.drawFrameAxes(frame_undistorted, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
            print(f"ID: {det.tag_id}, Rvec: {rvec.T}, Tvec: {tvec.T}")

    # Display the frame
    cv2.imshow('AprilTag Detection', frame_undistorted)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()