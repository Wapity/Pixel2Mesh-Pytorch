# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:07:20 2019

@author: lhuls
"""
from __future__ import division
import sys
import os
sys.path.append(os.getcwd() + '/..')
import definitions
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, name = None):
        super(Model, self).__init__()
        # allowed_kwargs = {'name', 'logging'}
        # for kwarg in kwargs.keys():
        #     assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        # name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        # logging = kwargs.get('logging', False)
        # self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.output1 = None
        self.output2 = None
        self.output3 = None
        self.output1_2 = None
        self.output2_2 = None

        self.loss = 0
        self.optimizer = None
        self.opt_op = None
    #end
        
    def build(self):
         # Build sequential resnet model
        eltwise = [3,5,7,9,11,13, 19,21,23,25,27,29, 35,37,39,41,43,45]
        concat = [15, 31]
        self.activations.append(self.inputs)
        for idx,layer in enumerate(self.layers):
            hidden = layer(self.activations[-1])
            if idx in eltwise:
                hidden = torch.add(hidden,self.activations[-2])*0.5
            if idx in concat:
                hidden = torch.concat([hidden, self.activations[-2]], 1)
            self.activations.append(hidden)

        self.output1 = self.activations[15]
        unpool_layer = GraphPooling(placeholders=self.placeholders, pool_id=1)
        self.output1_2 = unpool_layer(self.output1)

        self.output2 = self.activations[31]
        unpool_layer = GraphPooling(placeholders=self.placeholders, pool_id=2)
        self.output2_2 = unpool_layer(self.output2)

        self.output3 = self.activations[-1]

        # Store model variables for easy access
        variables = torch.get_collection(torch.GraphKeys.GLOBAL_VARIABLES,scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()

        self.opt_op = self.optimizer.minimize(self.loss)
    #end
        
           
#end
       
model = Model('test')

print(model.build())
        
        