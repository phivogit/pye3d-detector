import cv2
import numpy as np
import pyautogui
import argparse

class FrontCamThread:
    def __init__(self, cam_id, resolution):
        self.cam_id = cam_id
        self.resolution = resolution
        self.running = True

        self.camera_matrix = np.array([[343.34511283, 0.0, 327.80111243],
                                       [0.0, 342.79698299, 231.06509007],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([0, 0, 0, -0.001, -0.0])
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.screen_coords = None

    def run(self):
        cam = cv2.VideoCapture(self.cam_id)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        while self.running:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame from front camera")
                break

            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

            # Detect ArUco markers
            corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_parameters)
            if ids is not None and len(ids) == 4:
                # Get the screen coordinates from the ArUco markers
                self.screen_coords = self.get_screen_coords(corners, ids)
                if self.screen_coords is not None:
                    print("Perspective transform matrix computed")

            if self.screen_coords is not None:
                # Use the center point (320, 250) as the cursor position
                cursor_point = np.array([[[320, 250]]], dtype=np.float32)

                # Apply perspective transform
                transformed_point = cv2.perspectiveTransform(cursor_point, self.screen_coords)
                screen_x, screen_y = transformed_point[0][0]

                # Check if the point is within the screen bounds
                if 0 <= screen_x <= 1 and 0 <= screen_y <= 1:
                    print(f"Cursor point: (320, 250), Transformed: ({screen_x}, {screen_y})")
                    
                    # Map the normalized coordinates to screen resolution
                    screen_width, screen_height = pyautogui.size()
                    mouse_x = int(screen_x * screen_width)
                    mouse_y = int(screen_y * screen_height)
                    
                    pyautogui.moveTo(mouse_x, mouse_y)
                    
                    # For visualization
                    frame_points = cv2.perspectiveTransform(np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float32), np.linalg.inv(self.screen_coords))
                    cv2.polylines(frame, [frame_points.astype(int)], True, (0, 255, 0), 2)
                    
                    # Draw the ArUco markers
                    if ids is not None:
                        for corner in corners:
                            cv2.polylines(frame, [corner.astype(int)], True, (0, 0, 255), 2)
                    
                    # Draw the cursor point
                    cv2.circle(frame, (320, 250), 5, (255, 0, 0), -1)

            cv2.imshow("Front Camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

    def get_screen_coords(self, corners, ids):
        if len(corners) != 4:
            print(f"Detected {len(corners)} ArUco markers, expected 4.")
            return None

        # Initialize sorted_corners with None
        sorted_corners = [None, None, None, None]

        # Sort corners based on their IDs
        for corner, id in zip(corners, ids):
            if id[0] == 0:
                sorted_corners[0] = corner
            elif id[0] == 2:
                sorted_corners[1] = corner
            elif id[0] == 3:
                sorted_corners[2] = corner
            elif id[0] == 1:
                sorted_corners[3] = corner

        # Check if all markers were detected
        if any(corner is None for corner in sorted_corners):
            print("Not all required ArUco markers were detected.")
            return None

        # Get the outer corners of each marker
        tl = sorted_corners[0][0][0]  # Top-left corner of top-left marker (ID 0)
        tr = sorted_corners[1][0][1]  # Top-right corner of top-right marker (ID 2)
        br = sorted_corners[2][0][2]  # Bottom-right corner of bottom-right marker (ID 3)
        bl = sorted_corners[3][0][3]  # Bottom-left corner of bottom-left marker (ID 1)

        # Define the quadrilateral
        quad = np.array([tl, tr, br, bl], dtype=np.float32)

        # Define the actual screen coordinates of the markers
        screen_width, screen_height = pyautogui.size()
        actual_corners = np.array([
            [81, 51],              # Top-left (ID 0)
            [1838, 51],            # Top-right (ID 2)
            [1838, 1036],          # Bottom-right (ID 3)
            [81, 1036]             # Bottom-left (ID 1)
        ], dtype=np.float32)

        # Normalize the actual corners to [0, 1] range
        actual_corners[:, 0] /= screen_width
        actual_corners[:, 1] /= screen_height

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(quad, actual_corners)

        return M

    def stop(self):
        self.running = False

def main(args):
    front_cam_thread = FrontCamThread(args.front_cam, args.front_res)
    front_cam_thread.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Head-controlled cursor system")
    parser.add_argument("--front_cam", type=int, default=2, help="Front camera index")
    parser.add_argument("--front_res", nargs=2, type=int, default=[640, 480], help="Front camera resolution")
    args = parser.parse_args()
    
    main(args)