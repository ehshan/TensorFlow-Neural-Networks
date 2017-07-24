import tensorflow as tf

# values for nodes
T, F = 1., -1.
bias = 1.

# Training Set of 4 inputs
train_in = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias],
]

# Output patterns for input set
train_out = [
    [T],
    [F],
    [F],
    [F],
]

# Weights (3 x 1 tensor to match input patterns)
# Values randomly initialised
w = tf.Variable(tf.random_normal([3, 1]))

# Step Activation Function
# If x > 0 return 1 else return -1
def step(x):
    is_greater = tf.greater(x, 0)
    as_float = tf.to_float(is_greater)
    double = tf.multiply(as_float, 2)
    return tf.subtract(double, 1)