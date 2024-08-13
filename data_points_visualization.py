import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Step 1: Generate Imbalanced Sample Data
np.random.seed(42)

# Class 1 (500 points) and Class 2 (10,000 points) data
class1 = np.random.rand(500, 200) + np.array([0.5] * 200)
class2 = np.random.rand(10000, 200) + np.array([-0.5] * 200)

# Labels
labels_class1 = np.ones(500)  # Label 1 for Class 1
labels_class2 = np.zeros(10000)  # Label 0 for Class 2

# Combine data and labels
X = np.vstack((class1, class2))
y = np.hstack((labels_class1, labels_class2))

# Standardize the data (important for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)  # eps and min_samples can be tuned
dbscan_labels = dbscan.fit_predict(X_scaled)

# Number of clusters found (excluding noise)
num_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"Number of clusters found: {num_clusters}")

# Step 3: Validate Clusters
# Count the number of points from each class in each cluster
unique_clusters = set(dbscan_labels)
cluster_counts = {cluster: np.zeros(2) for cluster in unique_clusters}

for i in range(len(y)):
    cluster = dbscan_labels[i]
    true_class = int(y[i])
    if cluster != -1:  # Exclude noise points
        cluster_counts[cluster][true_class] += 1

print("\nCluster Counts (Clusters x Classes):")
for cluster, counts in cluster_counts.items():
    print(f"Cluster {cluster}: Class 0 count = {counts[0]}, Class 1 count = {counts[1]}")


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_classes=2, n_clusters_per_class=1, 
                           weights=[0.1, 0.9], flip_y=0, 
                           random_state=42)

# Initialize Random Forest Classifier with class weights
clf = RandomForestClassifier(class_weight='balanced', random_state=42)

# Fit the model
clf.fit(X, y)

# Predict on the same data
y_pred = clf.predict(X)

# Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Note: Noise points (label -1) are not included in the counts
noise_points = np.sum(dbscan_labels == -1)
print(f"\nNumber of noise points: {noise_points}")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Generating synthetic data
np.random.seed(0)
class_1 = np.random.randn(500, 200) + np.array([1] * 200)  # Class 1 centered at 1
class_2 = np.random.randn(500, 200) + np.array([3] * 200)  # Class 2 centered at 3

# Combining data into one dataset
X = np.vstack((class_1, class_2))
y = np.array([0] * 500 + [1] * 500)  # Labels: 0 for class 1, 1 for class 2

# Initialize LDA with 2 components
lda = LDA(n_components=2)

# Fit LDA and transform the data
X_lda = lda.fit_transform(X, y)

# Visualize the result
plt.figure(figsize=(8, 6))
plt.scatter(X_lda[y == 0, 0], X_lda[y == 0, 1], label='Class 1', alpha=0.7)
plt.scatter(X_lda[y == 1, 0], X_lda[y == 1, 1], label='Class 2', alpha=0.7)
plt.title('LDA: 200D to 2D Projection')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Generating synthetic data (as before)
np.random.seed(0)
class_1 = np.random.randn(500, 200) + np.array([1] * 200)  # Class 1 centered at 1
class_2 = np.random.randn(500, 200) + np.array([3] * 200)  # Class 2 centered at 3

# Combining data into one dataset
X = np.vstack((class_1, class_2))
y = np.array([0] * 500 + [1] * 500)  # Labels: 0 for class 1, 1 for class 2

# Initialize LDA with 1 component (since only 1 component is possible with 2 classes)
lda = LDA(n_components=1)

# Fit LDA and transform the data
X_lda = lda.fit_transform(X, y)

# Visualize the result in 1D
plt.figure(figsize=(8, 6))
plt.scatter(X_lda[y == 0], [0]*500, label='Class 1', alpha=0.7)
plt.scatter(X_lda[y == 1], [0]*500, label='Class 2', alpha=0.7)
plt.title('LDA: 200D to 1D Projection')
plt.xlabel('LDA Component 1')
plt.legend()
plt.show()