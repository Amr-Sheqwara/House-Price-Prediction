import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
############################################################
# Load the dataset
data = pd.read_csv('HousingData.csv')
# Fill NaN values
data.fillna(data.mean(), inplace=True)
# Select featuers(x1,x2,....), and target(Y)
X = data.iloc[: , :-1]
y = data['MEDV']
# Data splitting into (training & tsting data) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

############################################################
###################### LR implementation ###################
# Train the LR model
lr_sklearn = LinearRegression()
lr_sklearn.fit(X_train, y_train)
# Make predictions
y_pred_lr_sklearn = lr_sklearn.predict(X_test)
# Calculate MSE fot the model
mse_lr_sklearn = mean_squared_error(y_test, y_pred_lr_sklearn)

############################################################
###################### GB implementation ###################
# Train the GB model
gbr_sklearn = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr_sklearn.fit(X_train, y_train)
# Make predictions
y_pred_gbr_sklearn = gbr_sklearn.predict(X_test)
# Calculate MSE fot the model
mse_gbr_sklearn = mean_squared_error(y_test, y_pred_gbr_sklearn)

############################################################
# Print MSE
print(f"Linear Regression MSE: {mse_lr_sklearn}")
print(f"Gradient Boosting MSE: {mse_gbr_sklearn}")
# Print the diagram
plt.figure(figsize=(9, 6))
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.scatter(y_test, y_pred_gbr_sklearn, color='blue', label='Gradient Boosting')
plt.scatter(y_test, y_pred_lr_sklearn, color='red', label='Linear Regression')

plt.legend()
plt.show()