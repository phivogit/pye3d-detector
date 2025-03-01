import cv2
import numpy as np

# Camera calibration parameters from the previous calibration step
camera_matrix = np.array([[343.34511283, 0.0, 327.80111243],
                          [0.0, 342.79698299, 231.06509007],
                          [0.0, 0.0, 1.0]])
dist_coeffs = np.array([0 , 0, 0, -0.001])

# Initialize camera
camera_id = 2
cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print(f"Error: Cannot open camera with ID {camera_id}")
    exit()

# Disable autofocus
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0 = disable autofocus

# Set a fixed focus (optional, adjust the value as needed)
# cap.set(cv2.CAP_PROP_FOCUS, 0)  # 0 = focus at infinity

# Set up ArUco marker detection
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
parameters = cv2.aruco.DetectorParameters()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Undistort the frame
    frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Convert to grayscale
    gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Draw detected markers
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame_undistorted, corners, ids)
        for i in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)
            cv2.drawFrameAxes(frame_undistorted, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
            print(f"ID: {ids[i][0]}, Rvec: {rvec}, Tvec: {tvec}")

    # Display the frame
    cv2.imshow('ArUco Marker Detection', frame_undistorted)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()