import json
import numpy as np
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the data from JSON file
with open('eye_tracking_data2.json', 'r') as f:
    data = json.load(f)

# Extract features (gaze_direction) and targets (marker_position)
X = np.array([d['gaze_direction'] for d in data])
y = np.array([d['marker_position'] for d in data])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the MultiTaskLassoCV model
multi_task_lasso_cv = MultiTaskLassoCV(cv=5, random_state=42)
multi_task_lasso_cv.fit(X_train, y_train)

# Make predictions
y_pred = multi_task_lasso_cv.predict(X_test)

# Calculate the error for the predictions
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print the results
print("\nBest alpha value:")
print(multi_task_lasso_cv.alpha_)

print("\nMean Squared Error (MSE) for MultiTaskLasso Regression Predictions:")
print(mse)

print("\nRoot Mean Squared Error (RMSE) for MultiTaskLasso Regression Predictions:")
print(rmse)

# Calculate and print R-squared score
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
print("\nR-squared score (averaged over all outputs):")
print(r2)

# Print the number of features used by the model
n_features_used = np.sum(np.any(multi_task_lasso_cv.coef_ != 0, axis=0))
print(f"\nNumber of features used: {n_features_used}")

# Save the model
joblib.dump(multi_task_lasso_cv, 'multi_task_lasso_cv_model.joblib')

print("\nMultiTaskLassoCV model saved.")