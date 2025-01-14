import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# Load the data from JSON file
with open('eye_tracking_data.json', 'r') as f:
    data = json.load(f)

# Extract features (gaze_direction) and targets (marker_position)
X = np.array([d['gaze_direction'] for d in data])
y = np.array([d['marker_position'] for d in data])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the models
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Use NearestNeighbors to find the closest training point for each test point
nn_model = NearestNeighbors(n_neighbors=1)
nn_model.fit(X_train)
distances, indices = nn_model.kneighbors(X_test)

# Threshold to decide whether to use Random Forest or Linear Regression
threshold = 0.1  # This value can be tuned

# Make predictions
y_pred_combined = []

for i, dist in enumerate(distances):
    if dist < threshold:
        # If the test point is close to a training point, use Random Forest
        y_pred_combined.append(rf_model.predict([X_test[i]])[0])
    else:
        # Otherwise, use Linear Regression
        y_pred_combined.append(lr_model.predict([X_test[i]])[0])

y_pred_combined = np.array(y_pred_combined)

# Make predictions using individual models
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Calculate the error and R-squared for each model
mse_combined = mean_squared_error(y_test, y_pred_combined)
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_rf = mean_squared_error(y_test, y_pred_rf)

r2_combined = r2_score(y_test, y_pred_combined)
r2_lr = r2_score(y_test, y_pred_lr)
r2_rf = r2_score(y_test, y_pred_rf)

# Print the results
print("\nMean Squared Error (MSE):")
print(f"Combined: {mse_combined}")
print(f"Linear Regression: {mse_lr}")
print(f"Random Forest: {mse_rf}")

print("\nR-squared Score:")
print(f"Combined: {r2_combined}")
print(f"Linear Regression: {r2_lr}")
print(f"Random Forest: {r2_rf}")


