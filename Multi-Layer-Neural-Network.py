import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# generate sample data
np.random.seed(1)
data_points = 1000

# STATS FOR 3 CLUSTERS

# cluster means
c1_mean = [0, 0]
c2_mean = [1, 4]
c3_mean = [2, 8]
# cluster co-variance
c1_cov = [[2, .7], [.7, 2]]
c2_cov = [[1, .7], [.7, 1]]
c3_cov = [[0, .7], [.7, 0]]

# GENERATE DATA
# Cluster 1
c1 = np.random.multivariate_normal(c1_mean, c1_cov, data_points)
# Cluster 2
c2 = np.random.multivariate_normal(c2_mean, c2_cov, data_points)
# Cluster 3
c3 = np.random.multivariate_normal(c3_mean, c3_cov, data_points)

# Array holding all data for 3 clusters
data_features = np.vstack((c1, c2, c3)).astype(np.float32)

# Even distribution of labels -> 0, 1, 2
data_labels = np.hstack((np.zeros(data_points), np.ones(data_points), np.ones(data_points) + 1))

# VISUALISATION
# Set the graph size
plt.figure(figsize=(12, 8))
# Generate distribution of data points
plt.scatter(data_features[:, 0], data_features[:, 1], c=data_labels, alpha=.5)
# Show graph
plt.show()

# DATA NORMALISATION
# One-hot encoding for data labels
onehot_labels = np.zeros((data_labels.shape[0], 3)).astype(int)
onehot_labels[np.arange(len(data_labels)), data_labels.astype(int)] = 1

# split data to train/test
training_data, test_data, training_labels, test_labels = \
    train_test_split(data_features, onehot_labels, test_size=.1, random_state=12)

# NETWORK ARCHITECTURE 
# Assign Network Variables
hidden_nodes = 5
num_labels = training_labels.shape[1]
num_features = training_data.shape[1]
learning_rate = .01