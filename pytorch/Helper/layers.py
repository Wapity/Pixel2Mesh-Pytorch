#  Copyright (C) 2019 Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, Yu-Gang Jiang, Fudan University
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import division
from Helper.inits import *
import torch
import sys
import os
sys.path.append(os.getcwd() + '/..')
import definitions
import numpy as np


# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def project(img_feat, x, y, dim):
    x1 = torch.floor(x)
    #x2 = torch.ceil(x)
    x2 = torch.minimum(torch.ceil(x), torch.cast(torch.shape(img_feat)[0], torch.float32) - 1)
    y1 = torch.floor(y)
    #y2 = torch.ceil(y)
    y2 = torch.minimum(torch.ceil(y), torch.cast(torch.shape(img_feat)[1], torch.float32) - 1)
    Q11 = torch.gather_nd(img_feat, torch.stack([torch.cast(x1,torch.int32), torch.cast(y1,torch.int32)],1))
    Q12 = torch.gather_nd(img_feat, torch.stack([torch.cast(x1,torch.int32), torch.cast(y2,torch.int32)],1))
    Q21 = torch.gather_nd(img_feat, torch.stack([torch.cast(x2,torch.int32), torch.cast(y1,torch.int32)],1))
    Q22 = torch.gather_nd(img_feat, torch.stack([torch.cast(x2,torch.int32), torch.cast(y2,torch.int32)],1))

    weights = torch.multiply(torch.subtract(x2,x), torch.subtract(y2,y))
    Q11 = torch.multiply(torch.tile(torch.reshape(weights,[-1,1]),[1,dim]), Q11)

    weights = torch.multiply(torch.subtract(x,x1), torch.subtract(y2,y))
    Q21 = torch.multiply(torch.tile(torch.reshape(weights,[-1,1]),[1,dim]), Q21)

    weights = torch.multiply(torch.subtract(x2,x), torch.subtract(y,y1))
    Q12 = torch.multiply(torch.tile(torch.reshape(weights,[-1,1]),[1,dim]), Q12)

    weights = torch.multiply(torch.subtract(x,x1), torch.subtract(y,y1))
    Q22 = torch.multiply(torch.tile(torch.reshape(weights,[-1,1]),[1,dim]), Q22)

    outputs = torch.add_n([Q11, Q21, Q12, Q22])
    return outputs

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += torch.random_uniform(noise_shape)
    dropout_mask = torch.cast(torch.floor(random_tensor), dtype=torch.bool)
    pre_out = torch.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for torch.matmul (sparse vs dense)."""
    if sparse:
        res = torch.sparse_tensor_dense_matmul(x, y)
    else:
        res = torch.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
#         with torch.name_scope(self.name):
        if self.logging and not self.sparse_inputs:
            summary.histogram(self.name + '/inputs', inputs)
        outputs = self._call(inputs)
        if self.logging:
            torch.summary.histogram(self.name + '/outputs', outputs)
        return outputs

    def _log_vars(self):
        for var in self.vars:
            torch.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=False,
                 sparse_inputs=False, act=torch.nn.ReLU() , bias=True, gcn_block_id=1,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        if gcn_block_id == 1:
                self.support = placeholders['support1']
        elif gcn_block_id == 2:
                self.support = placeholders['support2']
        elif gcn_block_id == 3:
                self.support = placeholders['support3']
            
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = 3#placeholders['num_features_nonzero']

        with torch.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = torch.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = torch.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class GraphPooling(Layer):
    """Graph Pooling layer."""
    def __init__(self, placeholders, pool_id=1, **kwargs):
        super(GraphPooling, self).__init__(**kwargs)

        self.pool_idx = placeholders['pool_idx'][pool_id-1]

    def _call(self, inputs):    #might not be returning correct value as pool_idx wasnt used
        X = torch.tensor(inputs)  
        
        add_feat = (1/2.0) * torch.cumsum(X,0) #not sure if correct dimension
        
        # add_feat = (1/2.0) * tf.reduce_sum(tf.gather(X, self.pool_idx), 1)
        outputs = torch.cat([X, add_feat],0)
    
        return outputs

class GraphProjection(Layer):
    """Graph Pooling layer."""
    def __init__(self, placeholders, **kwargs):
        super(GraphProjection, self).__init__(**kwargs)

        self.img_feat = placeholders['img_feat']

    '''
    def _call(self, inputs):
        coord = inputs
        X = inputs[:, 0]
        Y = inputs[:, 1]
        Z = inputs[:, 2]

        #h = (-Y)/(-Z)*248 + 224/2.0 - 1
        #w = X/(-Z)*248 + 224/2.0 - 1 [28,14,7,4]
        h = 248.0 * torch.divide(-Y, -Z) + 112.0
        w = 248.0 * torch.divide(X, -Z) + 112.0

        h = torch.minimum(torch.maximum(h, 0), 223)
        w = torch.minimum(torch.maximum(w, 0), 223)
        indeces = torch.stack([h,w], 1)

        idx = torch.cast(indeces/(224.0/56.0), torch.int32)
        out1 = torch.gather_nd(self.img_feat[0], idx)
        idx = torch.cast(indeces/(224.0/28.0), torch.int32)
        out2 = torch.gather_nd(self.img_feat[1], idx)
        idx = torch.cast(indeces/(224.0/14.0), torch.int32)
        out3 = torch.gather_nd(self.img_feat[2], idx)
        idx = torch.cast(indeces/(224.0/7.00), torch.int32)
        out4 = torch.gather_nd(self.img_feat[3], idx)

        outputs = torch.concat([coord,out1,out2,out3,out4], 1)
        return outputs
    '''
    def _call(self, inputs):
        coord = inputs
        X = inputs[:, 0]
        Y = inputs[:, 1]
        Z = inputs[:, 2]

        h = 250 * torch.divide(-Y, -Z) + 112
        w = 250 * torch.divide(X, -Z) + 112

        h = torch.minimum(torch.maximum(h, 0), 223)
        w = torch.minimum(torch.maximum(w, 0), 223)

        x = h/(224.0/56)
        y = w/(224.0/56)
        out1 = project(self.img_feat[0], x, y, 64)

        x = h/(224.0/28)
        y = w/(224.0/28)
        out2 = project(self.img_feat[1], x, y, 128)

        x = h/(224.0/14)
        y = w/(224.0/14)
        out3 = project(self.img_feat[2], x, y, 256)

        x = h/(224.0/7)
        y = w/(224.0/7)
        out4 = project(self.img_feat[3], x, y, 512)
        outputs = torch.concat([coord,out1,out2,out3,out4], 1)
        return outputs
