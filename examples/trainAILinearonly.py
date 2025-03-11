import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load the data from JSON file
with open('eye_tracking_bucket1.json', 'r') as f:
    data = json.load(f)

# Extract features (gaze_direction) and targets (marker_position)
X = np.array([d['gaze_direction'] for d in data])
y = np.array([d['marker_position'] for d in data])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Calculate the error for the predictions
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)

# Print the results
print("\nMean Squared Error (MSE) for Linear Regression Predictions:")
print(mse)
print("\nRoot Mean Squared Error (RMSE) for Linear Regression Predictions:")
print(rmse)

# Calculate and print R-squared score
r2 = lr_model.score(X_test, y_test)
print("\nR-squared score:")
print(r2)

# Save the model
joblib.dump(lr_model, 'linearregressionmodelbucket3.joblib')

print("\nLinear Regression model saved.")