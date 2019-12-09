# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:41:46 2019

@author: lhuls

#Definitions for whole pytorch project.
"""

# Settings
learning_rate = 1e-5        #Initial learning rate.
epochs = 5                  #Number of epochs to train.
hidden = 256                #Number of units in hidden layer.
feat_dim = 963              #Number of units in feature layer.
coord_dim = 3               #Number of units in output layer.
weight_decay = 5e-6         #Weight decay for L2 loss.

data_file_path = 'Data/train_list.txt'
