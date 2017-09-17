import tensorflow as tf
import os
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

checkpoint_path = os.path.join("./tmp", "model.ckpt.meta")
print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=True)

#with tf.Session() as sess:

 # new_saver = tf.train.import_meta_graph('./tmp/model.ckpt.meta')
  #new_saver.restore(sess, './tmp')