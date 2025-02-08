import cv2
import threading
import argparse
import numpy as np
import torch
from pupil_detectors.detector_2d import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode

class ImprovedGazePredictionModel(torch.nn.Module):
    def __init__(self, hidden_size=256):
        super(ImprovedGazePredictionModel, self).__init__()
        
        self.input_size = 6
        self.output_size = 2
        
        # Improved gaze branch with residual connections
        self.gaze_branch = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(3, hidden_size//2),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_size//2),
                torch.nn.Dropout(0.1)
            ),
            torch.nn.Sequential(
                torch.nn.Linear(hidden_size//2, hidden_size//2),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_size//2),
                torch.nn.Dropout(0.1)
            )
        ])
        
        # Improved sphere branch with attention mechanism
        self.sphere_branch = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(3, hidden_size//2),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_size//2),
                torch.nn.Dropout(0.1)
            ),
            torch.nn.Sequential(
                torch.nn.Linear(hidden_size//2, hidden_size//2),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_size//2),
                torch.nn.Dropout(0.1)
            )
        ])
        
        # Attention mechanism for feature fusion
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 2),
            torch.nn.Softmax(dim=1)
        )
        
        # Main network with skip connections
        self.main_network = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_size),
                torch.nn.Dropout(0.2)
            ),
            torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size//2),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_size//2),
                torch.nn.Dropout(0.2)
            ),
            torch.nn.Linear(hidden_size//2, self.output_size)
        ])
        
    def forward(self, x):
        # Split input
        gaze_features = x[:, :3]
        sphere_features = x[:, 3:]
        
        # Process gaze branch with residual connection
        gaze_out = gaze_features
        for layer in self.gaze_branch:
            gaze_out = layer(gaze_out) + gaze_out if gaze_out.shape == layer(gaze_out).shape else layer(gaze_out)
            
        # Process sphere branch with residual connection
        sphere_out = sphere_features
        for layer in self.sphere_branch:
            sphere_out = layer(sphere_out) + sphere_out if sphere_out.shape == layer(sphere_out).shape else layer(sphere_out)
        
        # Combine features
        combined = torch.cat([gaze_out, sphere_out], dim=1)
        
        # Apply attention
        attention_weights = self.attention(combined)
        weighted_features = torch.stack([gaze_out, sphere_out], dim=2) * attention_weights.unsqueeze(1)
        attentive_features = weighted_features.sum(dim=2)
        
        # Process through main network with skip connections
        main_out = combined
        for layer in self.main_network[:-1]:
            main_out = layer(main_out) + main_out if main_out.shape == layer(main_out).shape else layer(main_out)
        
        return self.main_network[-1](main_out)

class SharedGazeData:
    def __init__(self):
        self.gaze_point = None
        self.confidence = 0.0
        self.lock = threading.Lock()

    def update(self, gaze_point, confidence):
        with self.lock:
            self.gaze_point = gaze_point
            self.confidence = confidence

    def get(self):
        with self.lock:
            return self.gaze_point, self.confidence

# Only showing the relevant modified sections - replace these in the previous code

class CamThread(threading.Thread):
    def __init__(self, preview_name, cam_id, resolution, is_eye_cam=False, focal_length=None, 
                 shared_gaze_data=None, camera_matrix=None, dist_coeffs=None):
        threading.Thread.__init__(self)
        self.preview_name = preview_name
        self.cam_id = cam_id
        self.resolution = resolution
        self.is_eye_cam = is_eye_cam
        self.shared_gaze_data = shared_gaze_data
        self.running = True
        self.debug_info = ""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        if is_eye_cam:
            self.detector_2d = Detector2D()
            self.camera = CameraModel(focal_length=focal_length, resolution=resolution)
            self.detector_3d = Detector3D(camera=self.camera, long_term_mode=DetectorMode.blocking)
            
            # Load the improved model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = ImprovedGazePredictionModel().to(self.device)
            
            try:
                # Load checkpoint
                checkpoint = torch.load('best_gaze_model2.pth', map_location=self.device)
                
                if 'model_state_dict' in checkpoint:
                    # If it's a training checkpoint
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # If it's just the state dict
                    self.model.load_state_dict(checkpoint)
                
                print("Successfully loaded model weights")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using untrained model - please train the model first")
            
            self.model.eval()
    def predict_gaze_point(self, gaze_normal, sphere_center, confidence):
        try:
            with torch.no_grad():
                # Normalize inputs
                sphere_center = np.array(sphere_center) / 20.0
                
                # Concatenate and convert to tensor
                features = np.concatenate([gaze_normal, sphere_center])
                input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                
                # Get prediction
                prediction = self.model(input_tensor)
                prediction = prediction.cpu().squeeze().numpy() * 500.0
                
                return tuple(map(int, prediction)), confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0

    def process_eye_frame(self, frame, frame_number, fps):
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result_2d = self.detector_2d.detect(grayscale)
        result_2d["timestamp"] = frame_number / fps
        result_3d = self.detector_3d.update_and_detect(result_2d, grayscale)
        
        if result_3d['confidence'] > 0.6 and 'circle_3d' in result_3d:
            gaze_normal = result_3d['circle_3d']['normal']
            sphere_center = result_3d['sphere']['center']
            
            gaze_point, confidence = self.predict_gaze_point(
                gaze_normal, sphere_center, result_3d['confidence']
            )
            
            if gaze_point is not None:
                self.shared_gaze_data.update(gaze_point, confidence)
                self.debug_info = f"Gaze: {gaze_point}, Conf: {confidence:.2f}"
            else:
                self.debug_info = "Invalid prediction"
        else:
            self.debug_info = "Low confidence or missing data"
        
        return result_3d

    def visualize_eye_result(self, frame, result_3d):
        if 'ellipse' in result_3d:
            ellipse = result_3d["ellipse"]
            cv2.ellipse(frame, 
                      tuple(int(v) for v in ellipse["center"]),
                      tuple(int(v / 2) for v in ellipse["axes"]),
                      ellipse["angle"], 0, 360, (0, 255, 0), 2)
            
            # Draw pupil center
            center = tuple(int(v) for v in ellipse["center"])
            cv2.circle(frame, center, 2, (0, 255, 255), -1)
            
        return frame

    def run(self):
        print(f'Starting {self.preview_name}')
        self.cam_preview()

    def stop(self):
        self.running = False

    def cam_preview(self):
        cam = cv2.VideoCapture(self.cam_id)
        if not cam.isOpened():
            print(f"Error: Could not open camera {self.cam_id}")
            return

        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        if not self.is_eye_cam:
            cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        fps = 60
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
                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

                gaze_point, confidence = self.shared_gaze_data.get()
                if gaze_point:
                    if 0 <= gaze_point[0] < frame.shape[1] and 0 <= gaze_point[1] < frame.shape[0]:
                        # Draw confidence-based visualization
                        radius = int(15 * confidence)
                        color_intensity = int(255 * confidence)
                        cv2.circle(frame, gaze_point, radius, (0, color_intensity, 255-color_intensity), -1)
                        cv2.circle(frame, gaze_point, radius, (255, 255, 255), 1)
                        self.debug_info = f"Gaze at: {gaze_point}, Conf: {confidence:.2f}"
                    else:
                        self.debug_info = f"Out of bounds: {gaze_point}"
                else:
                    self.debug_info = "No gaze point"

            # Add debug overlay
            cv2.putText(frame, self.debug_info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, self.debug_info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            cv2.imshow(self.preview_name, frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                break

            frame_count += 1

        cam.release()
        cv2.destroyWindow(self.preview_name)

def main(args):
    shared_gaze_data = SharedGazeData()

    # Camera calibration parameters
    camera_matrix = np.array([[343.34511283, 0.0, 327.80111243],
                             [0.0, 342.79698299, 231.06509007],
                             [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([0, 0, 0, -0.001, -0.0])

    eye_cam_thread = CamThread(
        "Eye Camera", args.eye_cam, args.eye_res, 
        is_eye_cam=True, 
        focal_length=args.focal_length, 
        shared_gaze_data=shared_gaze_data
    )
    
    front_cam_thread = CamThread(
        "Front Camera", args.front_cam, args.front_res, 
        shared_gaze_data=shared_gaze_data,
        camera_matrix=camera_matrix, 
        dist_coeffs=dist_coeffs
    )

    eye_cam_thread.start()
    front_cam_thread.start()

    try:
        eye_cam_thread.join()
        front_cam_thread.join()
    except KeyboardInterrupt:
        print("\nStopping threads...")
        eye_cam_thread.stop()
        front_cam_thread.stop()
        eye_cam_thread.join()
        front_cam_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced dual camera eye tracking system")
    parser.add_argument("--eye_cam", type=int, default=1, help="Eye camera index")
    parser.add_argument("--front_cam", type=int, default=2, help="Front camera index")
    parser.add_argument("--eye_res", nargs=2, type=int, default=[320, 240], 
                       help="Eye camera resolution")
    parser.add_argument("--front_res", nargs=2, type=int, default=[640, 480], 
                       help="Front camera resolution")
    parser.add_argument("--focal_length", type=float, default=84, 
                       help="Focal length of the eye camera")
    args = parser.parse_args()
    
    main(args)