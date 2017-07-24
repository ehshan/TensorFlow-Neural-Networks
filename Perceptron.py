import tensorflow as tf

# values for nodes
T, F = 1., -1.
bias = 1.

# Training Set of 4 inputs
training_in = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias],
]

# Output patterns for input set
training_out = [
    [T],
    [F],
    [F],
    [F],
]

# Weights (3 x 1 tensor to match input patterns)
# Values randomly initialised
weight = tf.Variable(tf.random_normal([3, 1]))


# Step Activation Function
# If x > 0 return 1 else return -1
def step(x):
    is_greater = tf.greater(x, 0)
    to_float = tf.to_float(is_greater)
    to_double = tf.multiply(to_float, 2)
    return tf.subtract(to_double, 1)


# Output function
output = step(tf.matmul(training_in, weight))

# Error function
error = tf.subtract(training_out, output)

# Mean Square Error
mse = tf.reduce_mean(tf.square(error))

# Calculating the weight adjustment determined by error
delta = tf.matmul(training_in, error, transpose_a=True)

# Assignment the adjustment value to the weight tensor
train = tf.assign(weight, tf.add(weight, delta))

# Create a TensorFlow session
session = tf.Session()
session.run(tf.initialize_all_variables())

# Initial error and the target value
err, target = 1, 0

# The number of epoch 0-10
epoch, max_epochs = 0, 10