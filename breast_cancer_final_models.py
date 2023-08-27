# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load your dataset (replace this with your dataset)
data = load_breast_cancer()
X = data.data
y = data.target
column_names = data.feature_names
# print(data)
# print(X.shape,y.shape)

# # Create a Random Forest Classifier (you can choose another classifier as well)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
#
# Get feature importances
feature_importances = clf.feature_importances_
print(feature_importances)
#
# Create a DataFrame to display feature importances (from the previous step)
importance_df = pd.DataFrame({'Feature': data.feature_names, 'Importance': feature_importances})

# Sort features by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)
# print(importance_df)
# # Select the top N most important features (you can change N)
top_features = importance_df['Feature'][:5].tolist()
print(top_features)
# print(importance_df)
# print("top_features",top_features)
# # Filter the dataset to keep only the selected features

X_selected = X[:, [data.feature_names.tolist().index(feature) for feature in top_features]]

# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Define a list of classifiers to use
classifiers = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Support Vector Machine', SVC(kernel='linear', C=1.0, random_state=42))
]
#
# Train and evaluate each classifier
for name, classifier in classifiers:
    print("".join(["-"]*100))
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("trained model: ",name)

    # Calculate validation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    # Compute the confusion matrix
    confusion = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(confusion)
    # Print or log the metrics for each classifier
    print(f"{name} Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("\n")
