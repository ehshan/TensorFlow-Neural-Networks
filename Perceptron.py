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