#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy
from PIL import Image
from scipy import ndimage

def create_placeholders(n_x, n_y):
	X = tf.placeholder(dtype = tf.float32, shape=(n_x, None))
	Y = tf.placeholder(dtype = tf.float32, shape=(n_y, None))
	return X, Y

def initialize_parameters(n_x, n_y):
	"""
	LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID
	W1: [25, n_X]
	b1: [25, 1]
	W2: [12, 25]
	b2: [12, 1]
	W3: [n_y, 12]
	b3: [n_y, 1]
	"""
	W1 = tf.get_variable('W1', [25, n_X], initializer = tf.contrib.layers.xavier_initializer())
	b1 = tf.get_variable('b1', [25, 1], initializer = tf.zeros_initializer())
	W2 = tf.get_variable('W2', [12, 25], initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable('b2', [12, 1], initializer = tf.zeros_initializer())
	W3 = tf.get_variable('W3', [n_y, 12], initializer = tf.contrib.layers.xavier_initializer())
	b3 = tf.get_variable('b3', [n_y, 1], initializer = tf.zeros_initializer())

	parameters = {'W1' : W1, 'b1' : b1, 'W2' : W2, 'b2' : b2, 'W3' : W3, 'b3' : b3}
	return parameters

def forward_propergation(X, parameters):
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	W3 = parameters['W3']
	b3 = parameters['b3']

	Z1 = tf.add(tf.matmul(W1, X), b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2, A1), b2)
	A2 = tf.nn.relu(Z2)
	Z3 = tf.add(tf.matmul(W3, A2), b3)
	return Z3

def compute_cost(Z3, Y):
	logits = tf.transpose(Z3)
	labels = tf.transpose(Y)
	cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
	return cost

def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1000, minibatch_size = 32):
	ops.reset_default_graph()
	(n_x, m) = X_train.shape
	n_y = Y_train.shape[0]
	costs = []
	X, Y = create_placeholders(n_x, n_y)
	parameters = initialize_parameters(n_x, n_y)
	Z3 = forward_propergation(X, parameters)
	cost = compute_cost(Z3, Y)
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).optimize(cost)
	init = tf.global_variables_initializer()
	with tf.Session as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			epoch_cost = 0
			num_minibatches = int(m / minibatch_size)
			minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

			for minibatch in minibatches:
				(minibatch_X, minibatch_Y) = minibatch
				_, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X:minibatch_X, Y:minibatch_Y})
				epoch_cost += minibatch_cost / num_minibatches
            if epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if epoch % 5 == 0:
                costs.append(epoch_cost)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
    return parameters

def main():
    img_path='/Users/james/Data/font_checker/positive_data/AaBuYu/uni4E1C_东.png'
    img = ndimage.imread(img_path, flatten=False)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

if __name__ == '__main__':
    main()
