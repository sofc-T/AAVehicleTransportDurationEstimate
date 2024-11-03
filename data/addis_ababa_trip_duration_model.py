# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load the training dataset
train_data = pd.read_csv('path/to/your/addis_ababa_trip_duration_train.csv')

# Define features and target variable
X = train_data.drop(columns=['Trip Duration (min)'])
y = train_data['Trip Duration (min)']

# Feature Engineering: Create new features
# Add binary feature for weekend
X['Is_Weekend'] = (X['Day of the Week'] == 'Weekend').astype(int)

# Drop the original 'Day of the Week' column
X = X.drop(columns=['Day of the Week'])

# Define categorical and numerical features
categorical_features = ['Weather', 'Is_Weekend']
numerical_features = ['Distance (km)', 'Start Hour']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Scale numerical features
        ('cat', OneHotEncoder(), categorical_features)   # One-hot encode categorical features
    ])

# Create a pipeline with Ridge regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))  # Alpha is the regularization strength
])

# Train the model with cross-validation
cross_val_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
mean_mse = np.mean(-cross_val_scores)

# Fit the model on the full training set
model.fit(X, y)

# Load the test dataset
test_data = pd.read_csv('path/to/your/addis_ababa_trip_duration_test.csv')

# Prepare the test features
X_test = test_data.drop(columns=['Trip Duration (min)'])
y_test = test_data['Trip Duration (min)']

# Test the model
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation results
print("Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print(f"Cross-Validated Mean MSE: {mean_mse:.2f}")
