from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from .inits import *


def project(img_feat, x, y, dim):
    ####### MAY BE WRONG
    #### TO CHECK
    img_size = img_feat.shape[-1]

    x = torch.clamp(x, min=0, max=img_size - 1)
    y = torch.clamp(y, min=0, max=img_size - 1)

    ### REASON FOR 7 without clamping is ERROR for GRAPHPROJECTION
    ### UNTIL HERE
    x1 = torch.floor(x).long()
    x2 = torch.ceil(x).long()
    y1 = torch.floor(y).long()
    y2 = torch.ceil(y).long()

    Q11 = img_feat[:, x1, y1].clone()
    Q12 = img_feat[:, x1, y2].clone()
    Q21 = img_feat[:, x2, y1].clone()
    Q22 = img_feat[:, x2, y2].clone()

    weights = torch.mul(x2.float() - x, y2.float() - y)
    #print(11)
    #print(weights.shape, weights.unsqueeze(-1).shape)
    #print(Q11.shape, torch.transpose(Q11, 0, 1).shape)
    #print(12)
    Q11 = torch.mul(weights.unsqueeze(-1), torch.transpose(Q11, 0, 1))

    weights = torch.mul(x2.float() - x, y - y1.float())
    Q12 = torch.mul(weights.unsqueeze(-1), torch.transpose(Q12, 0, 1))

    weights = torch.mul(x - x1.float(), y2.float() - y)
    Q21 = torch.mul(weights.unsqueeze(-1), torch.transpose(Q21, 0, 1))

    weights = torch.mul(x - x1.float(), y - y1.float())
    Q22 = torch.mul(weights.unsqueeze(-1), torch.transpose(Q22, 0, 1))

    outputs = sum([Q11, Q21, Q12, Q22])
    return outputs


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        #print('sparse', x.shape, y.shape)
        #res = torch.sparse.mm(x, y)
        res = torch.matmul(x.to_dense(), y)
    else:
        #print('dense', x.shape, y.shape)
        res = torch.matmul(x, y)
    return res


class GraphConvolution(nn.Module):
    """Graph convolution layer."""

    def __init__(self,
                 input_dim,
                 output_dim,
                 tensor_dict,
                 act=F.relu,
                 bias=True,
                 gcn_block_id=1,
                 featureless=False):
        super(GraphConvolution, self).__init__()
        self.layer_type = 'GraphConvolution'

        self.act = act
        if gcn_block_id == 1:
            self.support = tensor_dict['support1']
        elif gcn_block_id == 2:
            self.support = tensor_dict['support2']
        elif gcn_block_id == 3:
            self.support = tensor_dict['support3']

        self.featureless = featureless
        self.bias = bias
        self.vars = nn.ParameterDict()
        for i in range(len(self.support)):
            self.vars['weights_' + str(i)] = glorot([input_dim, output_dim])
        if self.bias:
            self.vars['bias'] = zeros([output_dim])

    def forward(self, inputs):
        x = inputs

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)], sparse=False)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = sum(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        if self.act is not None:
            output = self.act(output)
        return output


class GraphPooling(nn.Module):
    """Graph Pooling layer."""

    def __init__(self, tensor_dict, pool_id=1):
        super(GraphPooling, self).__init__()
        self.layer_type = 'GraphPooling'
        self.pool_idx = tensor_dict['pool_idx'][pool_id - 1]

    def forward(self, inputs):
        X = inputs.clone()
        add_feat = (1 / 2.0) * X[:, self.pool_idx].sum(2)
        output = torch.cat([inputs, add_feat], dim=1)
        return output


class GraphProjection(nn.Module):
    """Graph Projection layer."""

    def __init__(self):
        super(GraphProjection, self).__init__()
        self.layer_type = 'GraphProjection'

    def _prepare(self, img_feat):
        self.img_feat = img_feat

    def forward(self, inputs):
        outputs = []
        #print('inputs', inputs.shape)
        for idx, input_solo in enumerate(inputs):
            img_feat = [feats[idx] for feats in self.img_feat]
            output = self.forward_solo(inputs[0], img_feat)
            outputs.append(output)
        outputs = torch.stack(outputs, 0)
        return outputs

    def forward_solo(self, inputs, img_feat_solo):
        coord = inputs

        X = inputs[:, 0]
        Y = inputs[:, 1]
        Z = inputs[:, 2]

        h = 250 * torch.div(-Y, -Z) + 112
        w = 250 * torch.div(X, -Z) + 112

        h = torch.clamp(h, min=0, max=223)
        w = torch.clamp(w, min=0, max=223)

        x = h / (224.0 / 56)
        y = w / (224.0 / 56)

        out1 = project(img_feat_solo[0], x, y, 64)

        x = h / (224.0 / 28)
        y = w / (224.0 / 28)
        out2 = project(img_feat_solo[1], x, y, 128)

        x = h / (224.0 / 14)
        y = w / (224.0 / 14)
        out3 = project(img_feat_solo[2], x, y, 256)

        x = h / (224.0 / 7)
        y = w / (224.0 / 7)

        out4 = project(img_feat_solo[3], x, y, 512)

        #print('bllaaaaa')
        #for x in [coord, out1, out2, out3, out4]:
        #    print(x.shape)
        #output = torch.stack([coord, out1, out2, out3, out4], 0)

        outputs = torch.cat([coord, out1, out2, out3, out4], 1)
        #print('OUTPUT', outputs.shape)
        return outputs
