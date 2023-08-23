import numpy as np
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load your regression dataset (replace this with your dataset)
data = load_diabetes()
X = data.data
y = data.target

# Create a Random Forest Regressor (you can choose another regressor as well)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X, y)

# Get feature importances
feature_importances = regressor.feature_importances_

# Create a DataFrame to display feature importances
importance_df = pd.DataFrame({'Feature': data.feature_names, 'Importance': feature_importances})

# Sort features by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Select the top N most important features (you can change N)
top_features = importance_df['Feature'][:2].tolist()
print(importance_df)
print("top_features", top_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of regressors to use
regressors = [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', Ridge(alpha=1.0, random_state=42)),
    ('Lasso Regression', Lasso(alpha=1.0, random_state=42)),
    ('Elastic Net', ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)),
    ('Support Vector Regression', SVR(kernel='linear', C=1.0)),
    ('Random Forest Regressor', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('Gradient Boosting Regressor', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('XGBoost Regressor', XGBRegressor(random_state=42))
]

# Create empty lists to store metrics
models = []
mse_values = []
r2_values = []
# Train and evaluate each regressor
for name, regressor in regressors:
    print("".join(["_"]*50))
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print or log the metrics for each regressor
    print(f"{name} Metrics:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2): {r2}")
    print("\n")
    models.append(name)
    mse_values.append(mse)
    r2_values.append(r2)

# Create a DataFrame to display metrics
metrics_df = pd.DataFrame({'Model': models, 'MSE': mse_values, 'R2': r2_values})

# Print or display the metrics DataFrame
print(metrics_df)