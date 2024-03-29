import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

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

graph = tf.Graph()
with graph.as_default():
    # Generate data for TensorFlow
    tf_training_data = tf.placeholder(tf.float32, shape=[None, num_features])
    tf_training_labels = tf.placeholder(tf.float32, shape=[None, num_labels])
    tf_test_data = tf.constant(test_data)

    # Weights
    # Layer 1 array matches number of hidden nodes
    layer1_weights = tf.Variable(tf.truncated_normal([num_features, hidden_nodes]))
    # Layer 2 array matches number of labels
    layer2_weights = tf.Variable(tf.truncated_normal([hidden_nodes, num_labels]))

    # Biases for layers
    layer1_biases = tf.Variable(tf.zeros([hidden_nodes]))
    layer2_biases = tf.Variable(tf.zeros([num_labels]))


    # Three Layer Network Architecture
    def three_layer_network(data):
        input_layer = tf.matmul(data, layer1_weights)
        hidden_layer = tf.nn.relu(input_layer + layer1_biases)
        output_layer = tf.matmul(hidden_layer, layer2_weights) + layer2_biases
        return output_layer


    # Variable to hold model output
    model_scores = three_layer_network(tf_training_data)

    # Loss Function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_scores, labels=tf_training_labels))

    # Training set to minimise loss
    gradient_optimiser = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions - normalise output array to label probabilities - so sum of values=1
    train_prediction = tf.nn.softmax(model_scores)
    test_prediction = tf.nn.softmax(three_layer_network(tf_test_data))


# For labels compute the num of correct prediction over total predictions
def accuracy(predictions, labels):
    prediction_boolean = np.argmax(predictions, 1) == np.argmax(labels, 1)
    correct_predictions = np.sum(prediction_boolean)
    accuracy = 100.0 * correct_predictions / predictions.shape[0]
    return accuracy


# RUN THE NETWORK

# Number of iterations
epochs = 10000
# size of data partitions
batch_size = 100

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    for epoch in range(epochs):
        # Partition the training data
        offset = (epoch * batch_size) % (training_labels.shape[0] - batch_size)
        partition_data = training_data[offset:(offset + batch_size)]
        partition_labels = training_labels[offset:(offset + batch_size)]

        feed_dict = {tf_training_data: partition_data, tf_training_labels: partition_labels}

        _, l, predictions = session.run([gradient_optimiser, loss, train_prediction], feed_dict=feed_dict)

        if epoch % 1000 == 0:
            print(' Epoch: {0}: Mini-batch loss: {1}'.format(epoch, l))

    print('Test accuracy: {0}%'.format(accuracy(test_prediction.eval(), test_labels)))

    # Show number of correction and incorrect classifications
    labels_plot = np.argmax(test_labels, axis=1)
    t_predictions = np.argmax(test_prediction.eval(), axis=1)
    plt.figure(figsize=(12, 8))
    plt.scatter(test_data[:, 0], test_data[:, 1], c=t_predictions == labels_plot - 1, alpha=.5)
    plt.show()