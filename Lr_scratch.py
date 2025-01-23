######### Linear Regression From Scratch ##########
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Linear Regression Class
class LinearRegression:
    # Initialization the variables
    def __init__(self, learning_rate=1e-5, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    # Training the model
    def fit(self, X, y):
        # Initialize weights and bias as 0
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        # Training loop (Calculations to update weights and bias)
        for _ in range(self.n_iters):

            # Predict using y = X * weights + bias
            y_p = np.dot(X, self.weights) + self.bias

            # Calculate dw and db
            dw = (1 / n_samples) * np.dot(X.T, (y_p - y)) #dw
            db = (1 / n_samples) * np.sum(y_p - y)        #db

            # Update weights and bias
            self.weights -= self.lr * dw  # w = w - lr * dw
            self.bias -= self.lr * db     # b = b - lr * db

    # Testing the model
    def predict(self, x):
               # Y = X * weights + bias
        return np.dot(x, self.weights) + self.bias

#######################################################

# Mean Squared Error Calculation
def mse(y_test, predictions):
    return np.mean((y_test - predictions) ** 2)

#######################################################

# Load and preprocess data
data = pd.read_csv('HousingData.csv')
data.fillna(data.mean(), inplace=True)

# Extract features and target variable
X = data.iloc[: , :-1]
y = data['MEDV'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test =\
 train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
reg = LinearRegression(learning_rate=1e-6, n_iters=10000)
reg.fit(X_train, y_train)

# Predict on the test set
predictions = reg.predict(X_test)

# Calculate MSE
mse_value = mse(y_test, predictions)

######################################################

# Print Calculations

print("Performance Metric:")
print(f"Linear Regression from scratch MSE: {mse_value}")

#######################################################

# Visualization

plt.figure(figsize=(9, 5))
plt.scatter(y_test, predictions, color='blue', label='Linear Regression(from scratch)')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
