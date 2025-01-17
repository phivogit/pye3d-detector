import cv2
import numpy as np

def detect_display_region():
    # Camera calibration parameters
    camera_matrix = np.array([[343.34511283, 0.0, 327.80111243],
                            [0.0, 342.79698299, 231.06509007],
                            [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([0, 0, 0, -0.001, -0.0])

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        return

    kernel = np.ones((5,5), np.uint8)
    min_area = 640 * 480 * 0.1  # Pre-calculate minimum area

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Undistort frame using calibration parameters
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        
        # Convert to grayscale and threshold in one step
        _, white_regions = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 
                                      200, 255, cv2.THRESH_BINARY)
        
        # Clean up white regions
        white_regions = cv2.morphologyEx(white_regions, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(white_regions, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > min_area:
                approx = cv2.approxPolyDP(largest_contour, 
                                        0.02 * cv2.arcLength(largest_contour, True), 
                                        True)
                
                if len(approx) == 4:
                    # Sort corners
                    pts = np.float32(approx[:, 0])
                    rect = np.zeros((4, 2), dtype="float32")
                    
                    s = pts.sum(axis=1)
                    rect[0] = pts[np.argmin(s)]  # Top-left
                    rect[2] = pts[np.argmax(s)]  # Bottom-right
                    
                    diff = np.diff(pts, axis=1)
                    rect[1] = pts[np.argmin(diff)]  # Top-right
                    rect[3] = pts[np.argmax(diff)]  # Bottom-left
                    
                    # Draw minimal visualization
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                    for point in rect:
                        cv2.circle(frame, tuple(map(int, point)), 5, (0, 0, 255), -1)
        
        cv2.imshow('Display Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_display_region()