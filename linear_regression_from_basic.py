import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

# Load the Diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target.reshape(-1, 1)
# print(type(X),type(y))
# print(X.shape,y.shape,diabetes.target.shape)
# print(X[:10],y[:10],diabetes.target)

# Add a bias term (intercept)
X_b = np.c_[np.ones((X.shape[0], 1)), X]
# print(X_b[0])




#
# Linear Regression with Gradient Descent
def compute_cost(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def r_squared(y, y_pred):
    # Calculate the total sum of squares (TSS)
    tss = np.sum(np.square(y - np.mean(y)))
    # Calculate the residual sum of squares (RSS)
    rss = np.sum(np.square(y - y_pred))
    # Calculate R-squared (R^2)
    r2 = 1 - (rss / tss)
    return r2


def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []

    for _ in range(num_iterations):
        gradient = (1 / m) * X.T.dot(X.dot(theta) - y)
        theta -= learning_rate * gradient
        cost = compute_cost(theta, X, y)
        cost_history.append(cost)

    return theta, cost_history
#
# Initial values
theta = np.random.randn(X_b.shape[1], 1)  # Random initialization
learning_rate = 0.1
num_iterations = 20000




# Perform gradient descent to find optimal parameters
optimal_theta, cost_history = gradient_descent(X_b, y, theta, learning_rate, num_iterations)
print(optimal_theta,cost_history[0],cost_history[-1])
# # Print the optimal parameters (intercept and coefficients)
print("Optimal Parameters:")
print("Intercept (theta_0):", optimal_theta[0])
print("Coefficients (theta_1 to theta_10):", optimal_theta[1:])

# Calculate the Mean Squared Error (MSE) for the model
mse = compute_cost(optimal_theta, X_b, y)
print("Mean Squared Error (MSE):", mse)

# Make predictions on the dataset
y_pred = X_b.dot(optimal_theta)

# Calculate the R-squared (R^2) value
r2 = r_squared(y, y_pred)
print("R-squared (R^2) Value:", r2)
