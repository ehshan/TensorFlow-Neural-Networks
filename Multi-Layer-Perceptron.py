import tensorflow as tf

# values for nodes
T, F = 1., -1.

# Training Set of 4 input patterns
training_in = [
    [T, T],
    [T, F],
    [F, T],
    [F, F], ]

# Desired output patterns for an XOR
training_out = [
    [F],
    [T],
    [T],
    [F],
]

# Weights and biases for input layer
weight_1 = tf.Variable(tf.random_normal([2, 2]))
bias_1 = tf.Variable(tf.zeros([2]))

# Weights and biases for hidden layer
weight_2 = tf.Variable(tf.random_normal([2, 1]))
bias_2 = tf.Variable(tf.zeros([1]))


# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + tf.exp(-x))
