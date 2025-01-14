import cv2
import numpy as np

# Load the image
image = cv2.imread('C:/Users/hungn/Desktop/New folder/Capture6.PNG')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the dictionary that was used to generate the markers.
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

# Initialize the detector parameters using default values
parameters = cv2.aruco.DetectorParameters()

# Detect the markers in the image
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

# Draw detected markers
image_with_markers = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)

# Display the image with markers
cv2.imshow('Detected markers', image_with_markers)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the coordinates of the outer corners
if len(corners) == 4:
    for i, corner in enumerate(corners):
        print(f'Marker {ids[i][0]} corners:')
        for j in range(4):
            print(f'Corner {j}: {corner[0][j]}')
else:
    print('Could not detect exactly 4 markers')
