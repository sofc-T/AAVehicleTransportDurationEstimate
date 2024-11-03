# split_data.py

import pandas as pd
from sklearn.model_selection import train_test_split

# Load your processed dataset
data = pd.read_csv('./data/processed_data.csv')

# Separate features and target
X = data.drop('Trip Duration (min)', axis=1)  # Features
y = data['Trip Duration (min)']               # Target variable

# Split data into training+validation and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Further split training+validation into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=42)

# Save the splits in the data/splits folder
X_train.to_csv('data/splits/X_train.csv', index=False)
y_train.to_csv('data/splits/y_train.csv', index=False)
X_val.to_csv('data/splits/X_val.csv', index=False)
y_val.to_csv('data/splits/y_val.csv', index=False)
X_test.to_csv('data/splits/X_test.csv', index=False)
y_test.to_csv('data/splits/y_test.csv', index=False)

print("Data split and saved successfully!")
