import cv2
import numpy as np
import time

# Set up parameters for the chessboard
chessboard_size = (9, 6)
square_size = 1.0  # Adjust to your actual square size
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# Initialize camera
camera_id = 1
cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print(f"Error: Cannot open camera with ID {camera_id}")
    exit()

# Capture 30 images
num_images = 30
capture_interval = 0.75  # seconds
images_captured = 0

while images_captured < num_images:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)
        images_captured += 1
        print(f"Captured image {images_captured}/{num_images}")

    # Display the frame
    cv2.imshow('Calibration', frame)

    # Wait for 0.75 seconds
    time.sleep(capture_interval)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

if len(objpoints) > 0 and len(imgpoints) > 0:
    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("Camera calibration was successful.")
        print("Camera matrix:\n", camera_matrix)
        print("Distortion coefficients:\n", dist_coeffs)
    else:
        print("Camera calibration failed.")
else:
    print("Not enough images for calibration.")
