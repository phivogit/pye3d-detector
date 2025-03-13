import cv2
import numpy as np
import logging
from ctypes import windll
from pupil_apriltags import Detector as AprilTagDetector
from plugin import Plugin
import time

logger = logging.getLogger(__name__)

class Gaze_Control(Plugin):
    uniqueness = "by_class"
    order = 0.8
    
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.name = "Gaze Control"
        
        # Screen parameters (fixed at 1920x1080)
        self.screen_width = 1920
        self.screen_height = 1080
        
        # Gaze data
        self.current_gaze_point = None
        self.screen_transform = None
        self.status = "Initializing..."
        self.enabled = True
        
        # AprilTag detector
        self.apriltag_detector = AprilTagDetector(
            families='tag36h11',
            nthreads=1,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=True,
            decode_sharpening=0.25,
            debug=False
        )
        
        # Fixed marker corners on screen (1920x1080 coordinates)
        self.fixed_tag_corners = np.array([
            [49, 50],      # ID 0: Top-left corner (top-left of tag)
            [1869, 50],    # ID 1: Top-right corner (top-right of tag)
            [1874, 1035], # ID 3: Bottom-right corner (bottom-right of tag)
            [49, 1029]    # ID 2: Bottom-left corner (bottom-left of tag)
        ], dtype=np.float32)
        
        # Normalized screen surface definition
        self.screen_surface_definition = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1]
        ], dtype=np.float32)
        
        # Cursor smoothing and timing
        self.last_cursor_pos = (self.screen_width // 2, self.screen_height // 2)
        self.smoothed_cursor_pos = list(self.last_cursor_pos)  # For EMA
        self.last_update_time = time.time()
        self.cursor_update_interval = 0.033  # ~30 FPS
        self.smoothing_factor = 0.7  # Reduced smoothing for more responsiveness
        
    def detect_apriltags(self, frame):
        """Detect AprilTags in frame and compute screen transform using specific corners."""
        if frame is None:
            logger.debug("No frame provided for AprilTag detection")
            return None
            
        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.apriltag_detector.detect(gray)
        
        frame_height, frame_width = gray.shape
        
        if len(detections) != 4:
            logger.warning(f"Detected {len(detections)} tags, need exactly 4; stopping gaze control")
            return None  # Changed: Return None instead of using fallback
            
        tag_corners = {}
        for tag in detections:
            if tag.tag_id in [0, 1, 2, 3]:
                # Get specific corners based on tag ID
                # Corners order in pupil_apriltags: [bottom-left, bottom-right, top-right, top-left]
                if tag.tag_id == 0:  # Top-left corner (top-left of tag)
                    tag_corners[tag.tag_id] = tag.corners[3]  # top-left
                elif tag.tag_id == 1:  # Top-right corner (top-right of tag)
                    tag_corners[tag.tag_id] = tag.corners[2]  # top-right
                elif tag.tag_id == 3:  # Bottom-right corner (bottom-right of tag)
                    tag_corners[tag.tag_id] = tag.corners[1]  # bottom-right
                elif tag.tag_id == 2:  # Bottom-left corner (bottom-left of tag)
                    tag_corners[tag.tag_id] = tag.corners[0]  # bottom-left
                
        if len(tag_corners) != 4:
            logger.warning(f"Found {len(tag_corners)} valid tag corners; stopping gaze control")
            return None  # Changed: Return None instead of using fallback

        # Sort corners based on their IDs: ID 0 (top-left), ID 1 (top-right), ID 3 (bottom-right), ID 2 (bottom-left)
        quad = np.array([
            tag_corners[0],  # Top-left (ID 0)
            tag_corners[1],  # Top-right (ID 1)
            tag_corners[3],  # Bottom-right (ID 3)
            tag_corners[2]   # Bottom-left (ID 2)
        ], dtype=np.float32)

        # Normalize the fixed tag corners to [0, 1] range
        normalized_corners = self.fixed_tag_corners.copy()
        normalized_corners[:, 0] /= self.screen_width
        normalized_corners[:, 1] /= self.screen_height

        logger.debug(f"Detected quad: {quad}")
        logger.debug(f"Normalized corners: {normalized_corners}")
        
        return cv2.getPerspectiveTransform(quad, normalized_corners)
    
    def map_gaze_to_screen(self, gaze_point):
        """Map gaze point directly to screen coordinates (from pupilscreengaze.py)."""
        if self.screen_transform is None:
            return None
            
        # Convert gaze point to the format expected by perspectiveTransform
        point = np.array([[[gaze_point[0], gaze_point[1]]]], dtype=np.float32)
        
        try:
            # Apply perspective transformation
            transformed_point = cv2.perspectiveTransform(point, self.screen_transform)
            screen_x, screen_y = transformed_point[0][0]
            
            # Check if the point is within screen bounds (normalized coordinates)
            if 0 <= screen_x <= 1 and 0 <= screen_y <= 1:
                # Map normalized coordinates to screen resolution (fixed 1920x1080)
                mouse_x = int(screen_x * self.screen_width)
                mouse_y = int(screen_y * self.screen_height)
                return (mouse_x, mouse_y)
        except Exception as e:
            logger.error(f"Error mapping gaze to screen: {e}")
            
        return None
    
    def move_cursor(self, position):
        """Move cursor with EMA smoothing."""
        try:
            self.smoothed_cursor_pos[0] = (self.smoothing_factor * self.smoothed_cursor_pos[0] + 
                                          (1 - self.smoothing_factor) * position[0])
            self.smoothed_cursor_pos[1] = (self.smoothing_factor * self.smoothed_cursor_pos[1] + 
                                          (1 - self.smoothing_factor) * position[1])
            new_x, new_y = int(self.smoothed_cursor_pos[0]), int(self.smoothed_cursor_pos[1])
            windll.user32.SetCursorPos(new_x, new_y)
            self.last_cursor_pos = (new_x, new_y)
            return True
        except Exception as e:
            logger.error(f"Error moving cursor: {e}")
            return False
    
    def recent_events(self, events):
        """Process events efficiently with continuous AprilTag detection."""
        if not self.enabled:
            return
            
        current_time = time.time()
        
        # Update screen mapping on every frame
        frame = events.get('frame')
        if frame is not None:
            transform = self.detect_apriltags(frame.img)
            if transform is not None:
                self.screen_transform = transform
                self.status = "Screen mapping updated"
                logger.info("Screen mapping updated with AprilTags")
            else:
                self.screen_transform = None  # Changed: Explicitly reset transform if detection fails
                self.status = "Waiting for all 4 AprilTags to be detected"
                logger.warning("Screen mapping not updated; missing AprilTags")
        
        # Process gaze data with throttling, only if transform is valid
        if (current_time - self.last_update_time >= self.cursor_update_interval and 
            self.screen_transform is not None):
            gaze = events.get('gaze', [])
            if gaze:
                recent_gaze = gaze[-1]
                if recent_gaze.get('confidence', 0) < 0.6:
                    return
                    
                norm_pos = recent_gaze.get('norm_pos')
                if norm_pos is None:
                    return
                    
                # Use norm_pos as the gaze point (normalized [0, 1] in camera frame)
                # Convert to pixel coordinates based on frame size
                frame = events.get('frame')
                if frame is None:
                    return
                frame_height, frame_width = frame.img.shape[:2]
                gaze_pixel_x = norm_pos[0] * frame_width
                gaze_pixel_y = (1 - norm_pos[1]) * frame_height  # Invert y-coordinate
                self.current_gaze_point = (gaze_pixel_x, gaze_pixel_y)
                
                screen_point = self.map_gaze_to_screen(self.current_gaze_point)
                if screen_point:
                    self.move_cursor(screen_point)
            self.last_update_time = current_time
    
    def get_init_dict(self):
        return {'enabled': self.enabled}
    
    def cleanup(self):
        logger.info("Gaze Control plugin cleaned up")

def load_plugin(g_pool):
    return Gaze_Control(g_pool)