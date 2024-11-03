# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load the training dataset
train_data = pd.read_csv('../data/splits/X_train.csv')

# Define features and target variable
y = pd.read_csv('../data/splits/y_train.csv')
# print(y)
# print(train_data)

# train_data.rename(columns=lambda x: x.strip(), inplace=True)


print(train_data.columns)
X = train_data

# Feature Engineering: Create new features
# Add binary feature for weekend
X['Is_Weekend'] = (X['Day of the Week'] == 'Weekend').astype(int)

# Drop the original 'Day of the Week' column
X.drop(columns=['Day of the Week'], axis=1)

# Define categorical and numerical features
categorical_features = ['Weather_Clear', 'Weather_Cloudy', 'Weather_Rain', 'Is_Weekend']
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
test_data = pd.read_csv('../data/splits/X_test.csv')

# Prepare the test features
print(test_data.columns)
X_test = test_data
y_test = pd.read_csv('../data/splits/y_test.csv')


# add is weekend for the test data 

X_test['Is_Weekend'] = (X_test['Day of the Week'] == 'Weekend').astype(int)

# Test the model
y_pred = model.predict(X_test)

# Evaluate the model
mse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display evaluation results
print("Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print(f"Cross-Validated Mean MSE: {mean_mse:.2f}")
