import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

# Load the data from JSON file
with open('eye_tracking_data2.json', 'r') as f:
    data = json.load(f)

# Extract features (gaze_direction) and targets (marker_position)
X = np.array([d['gaze_direction'] for d in data])
y = np.array([d['marker_position'] for d in data])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the models
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Use NearestNeighbors to find the closest training point for each test point
nn_model = NearestNeighbors(n_neighbors=1)
nn_model.fit(X_train)
distances, indices = nn_model.kneighbors(X_test)

# Threshold to decide whether to use Decision Tree or Linear Regression
threshold = 0.1  # This value can be tuned

# Make predictions
y_pred_combined = []

for i, dist in enumerate(distances):
    if dist < threshold:
        # If the test point is close to a training point, use Decision Tree
        y_pred_combined.append(dt_model.predict([X_test[i]])[0])
    else:
        # Otherwise, use Linear Regression
        y_pred_combined.append(lr_model.predict([X_test[i]])[0])

y_pred_combined = np.array(y_pred_combined)

# Calculate the error for the combined predictions
mse_combined = mean_squared_error(y_test, y_pred_combined)

# Print the results
print("\nMean Squared Error (MSE) for Combined Predictions:")
print(mse_combined)

# Save the models
joblib.dump(lr_model, 'linear_regression_model.joblib')
joblib.dump(dt_model, 'decision_tree_model.joblib')
joblib.dump(nn_model, 'nearest_neighbors_model.joblib')

# Save the threshold
with open('threshold.txt', 'w') as f:
    f.write(str(threshold))

print("\nModels and threshold saved.")

