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
import torch
import numpy as np

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = torch.randint(shape,low=-scale,high=scale, dtype=torch.float32 )
    return initial


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = torch.randint(shape,low=-init_range,high=init_range,dtype=torch.float32)
    return initial


def zeros(shape, name=None):
    """All zeros."""
    initial = torch.zeros(shape,dtype=torch.float32)
    return initial


def ones(shape, name=None):
    """All ones."""
    initial = torch.ones(shape,dtype=torch.float32)
    return initial
