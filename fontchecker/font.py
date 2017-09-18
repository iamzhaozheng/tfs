#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import os 
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import ndimage
from PIL import Image

def loadfont(basedir):
    dirs = glob.glob(basedir + '/Aa*')
    names = [os.path.basename(x)[2:] for x in dirs]
    fonts = {}
    for i in range(0, len(dirs)):
        paths = glob.glob(x + '/uni*')
        fonts[names[i]] = paths
        print('Font loaded: ' + names[i])
    return fonts

def load_dataset(fonts, positive = 0):
    X = []
    for font, paths in fonts.iteritems():
        print('********** Loading ' + font + ' **********')
        for path in paths:
            img = np.array(ndimage.imread(path, flatten = False))
            img_flatten = img.reshape(1, 128 * 128)
            X.append(img_flatten)
    if (positive == 0):
        Y = np.zeros((len(fonts), 1))
    else:
        Y = np.ones((len(fonts), 1))
    return X, Y

def getMaxLength(fonts):
    maxl = 0
    for font, paths in fonts.iteritems():
        print('********** ' + font + ' **********')
        for path in paths:
            #img = ndimage.imread(path)
            #height, width = ndimage.imread(path).shape
            with Image.open(path) as img:
                width, height = img.size
                maxl = max(maxh, height)
                maxl = max(maxw, width)
    print('max length: ' + maxl)
    return maxl

def preprocess(fonts, length, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for font, paths in fonts.iteritems():
        outfontdir = os.path.join(outdir, 'Aa' + font)
        print(outfontdir)
        if not os.path.exists(outfontdir):
            os.makedirs(outfontdir)     
        for path in paths:
            img = Image.open(path)
            size = (length, length)
            img.thumbnail(size)
            img_padded = Image.new(img.mode, size, color = 255)
            img_padded.paste(img,(int((size[0]-img.size[0])/2),int((size[1]-img.size[1])/2)))
            directory, filename = os.path.split(path)
            filename_out = filename.replace('uni', 'cln', 1)
            tmp = os.path.join(outfontdir, filename_out)
            img_padded.save(tmp)
        print(font + ' preprocessed.')

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
    #image_file = '/Users/jameszhao/projects/data/training_data/positive_data/AaYaotiaoshunv/uni8E92_躒.png'
    #img = ndimage.imread(image_file, flatten=False)
    #plt.imshow(img, cmap='gray')
    #plt.show()
    #pos_dir = '/Users/james/Data/font_checker/positive_data'
    #cln_pos_dir = '/Users/james/Data/font_checker/clean_positive_data'
    #preprocess(fonts, 128, cln_pos_dir)
    fonts = loadfont(cln_pos_dir)


if __name__=='__main__':
    main()