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

# Note: Noise points (label -1) are not included in the counts
noise_points = np.sum(dbscan_labels == -1)
print(f"\nNumber of noise points: {noise_points}")
