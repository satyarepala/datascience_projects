import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Create class imbalance manually
class0_indices = np.where(y == 0)[0]
class1_indices = np.where(y == 1)[0]

# Select a subset of samples for each class to create imbalance
selected_indices = np.concatenate((
    class0_indices[:10],  # Select 50 samples from class 0
    class1_indices[:]      # Keep all samples from class 1
))

X_imbalanced = X[selected_indices]
y_imbalanced = y[selected_indices]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imbalanced, y_imbalanced, test_size=0.2, random_state=42)

# Create a Random Forest Classifier (you can choose another classifier as well)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate validation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion)


# Assuming you have already trained your classifier and have predictions (y_true and y_pred)

# Calculate precision and recall for class 0
precision_class0 = precision_score(y_test, y_pred, pos_label=0)
recall_class0 = recall_score(y_test, y_pred, pos_label=0)

# Calculate precision and recall for class 1
precision_class1 = precision_score(y_test, y_pred, pos_label=1)
recall_class1 = recall_score(y_test, y_pred, pos_label=1)

# Print the results
print("Class 0 Precision:", precision_class0)
print("Class 0 Recall:", recall_class0)
print("Class 1 Precision:", precision_class1)
print("Class 1 Recall:", recall_class1)

