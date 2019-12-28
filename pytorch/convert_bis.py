import tensorflow as tf
print(tf.__version__)
import os

sess = tf.Session()
saver = tf.train.import_meta_graph(os.getcwd() + '/tf_checkpoint/gcn.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint(os.getcwd() + '/tf_checkpoint/'))

graph = tf.get_default_graph()

x = graph.get_tensor_by_name('gcn/graphconvolution_7_vars/bias:0')
print(x)
