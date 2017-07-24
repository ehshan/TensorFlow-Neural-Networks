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
# Values initial to random
w = tf.Variable(tf.random_normal([3, 1]))