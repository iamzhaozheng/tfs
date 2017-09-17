import tensorflow as tf

with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('./tmp/word2vec_model.meta')
  new_saver.restore(sess, './')