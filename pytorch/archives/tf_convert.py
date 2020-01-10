import os
from pprint import pprint
import torch
import tensorflow as tf
import numpy as np
from collections import OrderedDict
tf_path = os.path.abspath(
    './tf_checkpoint/gcn.ckpt')  # Path to our TensorFlow checkpoint
tf_vars = tf.train.list_variables(tf_path)

# from tensorflow.python import pywrap_tensorflow
#
# reader = pywrap_tensorflow.NewCheckpointReader(tf_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
#
# print(var_to_shape_map)
# print(100 * '______')

# def listing():
#     c = tf.train.Checkpoint()
#     m = tf.train.CheckpointManager(c,
#                                    os.path.abspath('./tf_checkpoint'),
#                                    max_to_keep=2)
#     p = m.latest_checkpoint
#     vs = tf.train.list_variables(p)
#     print(f'names and shapes list: {vs}')
#     print('blaaaaa')
#     n, _ = vs[-1]
#     v = tf.train.load_variable(p, n)
#     print(f'loaded value: {v} for name: {n}')
#     c = tf.train.load_checkpoint(p)
#     ts = c.get_variable_to_dtype_map()
#     ss = c.get_variable_to_shape_map()
#     print(f'checkpoint types: {ts} and shapes: {ss}')

#listing()


def get_name(var):
    name = var[0]
    if name.startswith('gcn/Conv2D'):
        x = 0
    else:
        x = 1000
    if '_' in name:
        number = name.split('_')[1]
        number = int(number.split('/')[0])
        number += x
    else:
        number = x
    if 'bias' in var[0]:
        number += 0.9
    return number


tf_vars.sort(key=get_name)
# for tf_name, tf_shape in tf_vars[::-1]:
#     print(tf_name)
#     print(os.path.abspath('./tf_checkpoint/gcn.ckpt'))
#     tf.train.load_variable(os.path.abspath('./tf_checkpoint/gcn.ckpt'), tf_name)


def load(model):
    idx = 0
    state_dict = OrderedDict()
    for name, param in model.state_dict().items():
        if not 'str' in name:
            tf_name, tf_shape = tf_vars[idx]

            save_name = tf_name.replace('/', '|').replace(':', '=')
            path = os.path.abspath('./weights/') + '/' + save_name + '.npy'
            array = np.load(path)
            torch_tensor = torch.from_numpy(array)
            if len(tf_shape) == 4:
                torch_tensor = torch_tensor.permute(3, 2, 0, 1)
            state_dict[name] = torch_tensor
            #print('Torch : ', list(param.shape), '    - Tensorflow : ', tf_shape)
            print('Loading variable {} with shape {}'.format(tf_name, tf_shape))
            idx += 1
        else:
            state_dict[name] = param
    model.load_state_dict(state_dict)
    print('Model loaded from tensorflow')


# Load tensorflow model and save in numpy
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.allow_soft_placement = True
# sess = tf.Session(config=config)
# sess.run(tf.global_variables_initializer())
# model.load(sess)
# # all_vars = dict()
# # for name, tensor in model.vars.items():
# #     array = sess.run(tensor)
# #     print(name, array.shape)
# #     save_name = name.replace('/', '|').replace(':', '=')
# #     path = os.path.abspath('./weights/') + '/' + save_name
# #     np.save(path, array)
# #Runing the demo
