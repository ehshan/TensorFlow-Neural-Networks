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


# Bias can be adjusted so form part of the output equation
output1 = sigmoid(tf.add(tf.matmul(training_in, weight_1), bias_1))

output2 = sigmoid(tf.add(tf.matmul(output1, weight_2), bias_2))

# error function
error = tf.subtract(training_out, output2)

# Mean Square Error
mse = tf.reduce_mean((tf.square(error)))

# Training set to minimise the mean square error (data set -1 to 1) -  optimised on gradient decent
train = tf.train.GradientDescentOptimizer(0.01).minimize(mse)

# Create a TensorFlow session
session = tf.Session()
session.run(tf.initialize_all_variables())

# Initial error and the target value
err, target = 1.0, 0.1

# The number of epoch 0-10000
epoch, max_epoch = 0, 10000

# Main training function
while err > target and epoch < max_epoch:
    epoch += 1
    err, _ = session.run([mse, train])
    if (epoch % 1000) == 0:
        print('epoch:', epoch, 'mse:', err)