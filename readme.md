
# Addis Ababa Vehicle Transport Trip Duration Predictor
# [Report](https://github.com/sofc-T/AAVehicleTransportDurationEstimate/blob/main/reports/model_verfication.ipynb)
## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Model Details](#model-details)

## Introduction
This project aims to predict the duration of vehicle transport trips in Addis Ababa based on various features such as distance, start hour, day of the week, and weather conditions. The model is built using linear regression techniques and can be used to estimate trip durations for better transportation planning2.

- Linear Regression used in the report notebook
- Ridge regression used in the model training note book to penalize/ regularize larger Ceofficients and reduce overfitting - better prediction.

## Requirements
- Python 3.x
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - statsmodels


## Setup
You can install the required libraries using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels
Setup
Clone the Repository:

bash
Copy code
git clone <repository_url>
cd Addis_Trip_Duration_Predictor
Prepare the Dataset: Place your dataset in the data/ directory. Ensure it's in the CSV format and named processed_data.csv.
```

## usage
```
Split the Data: Run the split_data.py script to prepare training, validation, and test sets:

bash
Copy code
python split_data.py
Usage
Run the Model Training Notebook:

Open the notebooks/model_training.ipynb file in Jupyter Notebook or any compatible environment.
Execute the cells sequentially to train the model and evaluate its performance.
Make Predictions: After training the model, you can make predictions using the trained model. Here's a quick example:

python
Copy code
# Load the model and make predictions
import pandas as pd
from sklearn.linear_model import LinearRegression
```

## Model-details

# Make predictions

```
Algorithm: Linear Regression
Features Used:
Distance (km)
Start Hour
Day of the Week
Weather Condition
Evaluation Metrics:
Mean Squared Error (MSE)
R² Score
```
