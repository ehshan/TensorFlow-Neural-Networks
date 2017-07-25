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
