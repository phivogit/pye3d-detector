import json
import numpy as np
from typing import List, Dict, Any

class EyeTrackingNormalizer:
    def __init__(self, image_width: int = 640, image_height: int = 480):
        self.image_width = image_width
        self.image_height = image_height

    def normalize_marker_position(self, position: List[float]) -> List[float]:
        """Normalize marker position from pixel coordinates to [-1, 1] range."""
        x_norm = (2 * position[0] / self.image_width) - 1
        y_norm = (2 * position[1] / self.image_height) - 1
        return [x_norm, y_norm]

    def normalize_sphere_centers(self, data: List[Dict[str, Any]]) -> List[List[float]]:
        """Normalize sphere centers using statistics from the entire dataset."""
        # Extract all sphere centers
        sphere_centers = np.array([d['sphere_center'] for d in data])
        
        # Calculate statistics for normalization
        x_max = np.max(np.abs(sphere_centers[:, 0]))
        y_max = np.max(np.abs(sphere_centers[:, 1]))
        z_mean = np.mean(sphere_centers[:, 2])
        z_std = np.std(sphere_centers[:, 2])

        # Normalize each sphere center
        normalized_centers = []
        for center in sphere_centers:
            x_norm = center[0] / x_max if x_max != 0 else 0
            y_norm = center[1] / y_max if y_max != 0 else 0
            z_norm = (center[2] - z_mean) / z_std if z_std != 0 else 0
            normalized_centers.append([x_norm, y_norm, z_norm])

        return normalized_centers

    def normalize_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize the entire dataset."""
        # First normalize sphere centers for the entire dataset
        normalized_sphere_centers = self.normalize_sphere_centers(data)

        normalized_data = []
        for idx, entry in enumerate(data):
            normalized_entry = {
                'frame': entry['frame'],
                'confidence': entry['confidence'],
                'gaze_direction': entry['gaze_direction'],  # Already normalized
                'marker_position': self.normalize_marker_position(entry['marker_position']),
                'sphere_center': normalized_sphere_centers[idx]
            }
            normalized_data.append(normalized_entry)

        return normalized_data

def normalize_and_save(input_file: str, output_file: str, image_width: int = 640, image_height: int = 480):
    """Load JSON data, normalize it, and save to a new file."""
    try:
        # Read input JSON file
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Create normalizer and normalize data
        normalizer = EyeTrackingNormalizer(image_width, image_height)
        normalized_data = normalizer.normalize_data(data)

        # Save normalized data to output file
        with open(output_file, 'w') as f:
            json.dump(normalized_data, f, indent=2)

        print(f"Successfully normalized data and saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {input_file}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

# Example usage
if __name__ == "__main__":
    normalize_and_save(
        input_file="eye_tracking_data2.json",
        output_file="normalized_data.json",
        image_width=640,
        image_height=480
    )