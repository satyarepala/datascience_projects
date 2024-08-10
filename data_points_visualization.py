import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Step 2: Apply k-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
cluster_labels = kmeans.labels_

# Step 3: Validate Clusters
# Count the number of points from each class in each cluster
cluster_counts = np.zeros((2, 2))  # Shape: (number of clusters, number of classes)

for i in range(len(y)):
    cluster = cluster_labels[i]
    true_class = int(y[i])
    cluster_counts[cluster, true_class] += 1

print("Cluster Counts (Clusters x Classes):")
print(cluster_counts)

# Step 4: Optional - 3D Visualization of Clusters
# Reduce dimensionality to 3D for visualization
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting clusters
scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=cluster_labels, cmap='viridis', marker='o')

# Plotting class labels
for label in np.unique(y):
    ax.scatter(X_reduced[y == label, 0], X_reduced[y == label, 1], X_reduced[y == label, 2], label=f'Class {int(label)}')

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
plt.title('k-Means Clustering (k=2)')
plt.legend()
plt.show()
