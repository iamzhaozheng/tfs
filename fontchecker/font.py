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

def main():
    img_path='/Users/james/Data/font_checker/positive_data/AaBuYu/uni4E1C_东.png'
    img = ndimage.imread(img_path, flatten=False)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

if __name__ == '__main__':
    main()
