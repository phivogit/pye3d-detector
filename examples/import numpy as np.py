import numpy as np
from tensorflow import keras

def load_model(model_path):
    try:
        custom_objects = {
            'mse': keras.losses.MeanSquaredError(),
            'mae': keras.losses.MeanAbsoluteError(),
        }
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def parse_input(input_str):
    # Remove any parentheses and split by comma
    clean_str = input_str.replace('(', '').replace(')', '').strip()
    values = clean_str.split(',')
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got {len(values)}")
    return [float(v.strip()) for v in values]

def get_user_input():
    while True:
        try:
            sphere_center_input = input("Enter sphere center (x,y,z): ")
            sphere_center = parse_input(sphere_center_input)
            
            gaze_direction_input = input("Enter gaze direction (x,y,z): ")
            gaze_direction = parse_input(gaze_direction_input)
            
            return np.array(sphere_center).reshape(1, 3), np.array(gaze_direction).reshape(1, 3)
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

def predict_gaze_point(model, sphere_center, gaze_direction):
    try:
        prediction = model.predict([sphere_center, gaze_direction])[0]
        return tuple(map(int, prediction))
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def main():
    model_path = 'deep_learning_gaze_model_no_norm.h5'
    model = load_model(model_path)
    
    if model is None:
        return
    
    while True:
        sphere_center, gaze_direction = get_user_input()
        print(f"Sphere center: {sphere_center.flatten()}")
        print(f"Gaze direction: {gaze_direction.flatten()}")
        
        gaze_point = predict_gaze_point(model, sphere_center, gaze_direction)
        
        if gaze_point is not None:
            print(f"Predicted gaze point: {gaze_point}")
        else:
            print("Failed to predict gaze point.")
        
        continue_running = input("Do you want to make another prediction? (y/n): ").lower()
        if continue_running != 'y':
            break

if __name__ == "__main__":
    main()