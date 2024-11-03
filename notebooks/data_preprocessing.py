# data_preprocessing.ipynb

import pandas as pd

# Load raw data
data = pd.read_csv('../data/addis_ababa_trip_duration_train.csv')

# Perform data cleaning and preprocessing 
data['Day of the Week'] = data['Day of the Week'].map({'Weekday': 0, 'Weekend': 1})
data = pd.get_dummies(data, columns=['Weather'])

# Save the cleaned data to processed_data.csv
data.to_csv('../data/processed_data.csv', index=False)

print("Data cleaned and saved successfully!")
