# Linearity check
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load training data
X_train = pd.read_csv('../data/splits/X_train.csv')
y_train = pd.read_csv('../data/splits/y_train.csv')

# Check for linearity with scatter plots
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train['Distance (km)'], y=y_train.squeeze())
plt.xlabel('Distance (km)')
plt.ylabel('Trip Duration (min)')
plt.title('Distance vs. Trip Duration')
plt.show()


# Independence check    
# Correlation matrix for the features in X_train
plt.figure(figsize=(10, 6))
sns.heatmap(X_train.corr(), annot=True, cmap="coolwarm")
plt.title('Feature Correlation Matrix')
plt.show()


# Check for outliers in Trip Duration
plt.figure(figsize=(10, 6))
sns.boxplot(y=y_train.squeeze())
plt.title('Boxplot of Trip Duration')
plt.ylabel('Trip Duration (min)')
plt.show()



# Histogram of Trip Duration
plt.figure(figsize=(10, 6))
sns.histplot(y_train.squeeze(), bins=20, kde=True)
plt.title('Distribution of Trip Duration')
plt.xlabel('Trip Duration (min)')
plt.ylabel('Frequency')
plt.show()


# Histogram of Trip Duration
plt.figure(figsize=(10, 6))
sns.histplot(y_train.squeeze(), bins=20, kde=True)
plt.title('Distribution of Trip Duration')
plt.xlabel('Trip Duration (min)')
plt.ylabel('Frequency')
plt.show()


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

print(vif_data)




from sklearn.linear_model import LinearRegression

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_val)  # Use validation set for predictions



residuals = y_val.squeeze() - y_pred

# Residuals vs. Fitted Values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.show()

# Histogram of Residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=20, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()




from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate performance metrics
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")


from sklearn.linear_model import Ridge

# Train a Ridge regression model with regularization
ridge_model = Ridge(alpha=1.0)  # Modify alpha as needed
ridge_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred_ridge = ridge_model.predict(X_val)
mse_ridge = mean_squared_error(y_val, y_pred_ridge)
print(f"Ridge Mean Squared Error: {mse_ridge:.2f}")



# Plot predicted vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('Actual Trip Duration (min)')
plt.ylabel('Predicted Trip Duration (min)')
plt.title('Actual vs. Predicted Trip Duration')
plt.show()


# Calculate residuals
residuals = y_val.squeeze() - y_pred

# Residuals vs. Fitted Values Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.show()

# Histogram of Residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=20, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Q-Q Plot for Residuals
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot for Residuals')
plt.show()

