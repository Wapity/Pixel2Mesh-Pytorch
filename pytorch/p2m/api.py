from __future__ import division
import torch
import torch.nn as nn
from .inits import *
from .layers import *
from .utils import *
import tensorflow as tf


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.layers = []

    def _prepare(self, img_feat):
        for layer in self.proj_layers:
            layer._prepare(img_feat)

    def _build(self):
        raise NotImplementedError

    def build(self):
        self._build()
        self.unpool_layers = []
        self.unpool_layers.append(
            GraphPooling(tensor_dict=self.tensor_dict, pool_id=1))
        self.unpool_layers.append(
            GraphPooling(tensor_dict=self.tensor_dict, pool_id=2))

    def forward(self, img_inp, img_bis=None):
        reshape = len(img_inp.shape) == 3
        if reshape:
            img_inp = img_inp.unsqueeze(0)
        inputs = get_features(self.tensor_dict, img_inp)

        if self.args.cnn_type != 'STR':
            img_feat = self.forward_cnn(img_inp)
        else:
            img_feat = self.forward_cnn(img_inp, img_bis)
        self._prepare(img_feat)

        # Build sequential resnet model
        eltwise = [
            3, 5, 7, 9, 11, 13, 19, 21, 23, 25, 27, 29, 35, 37, 39, 41, 43, 45
        ]
        concat = [15, 31]

        activations = []
        activations.append(inputs)

        for idx, layer in enumerate(self.layers):
            hidden = layer(activations[-1])
            if idx in eltwise:
                hidden = torch.add(hidden, activations[-2]) * 0.5
            if idx in concat:
                hidden = torch.cat([hidden, activations[-2]], 2)
            activations.append(hidden)

        output1 = activations[15]
        output1_2 = self.unpool_layers[0](output1)

        output2 = activations[31]
        output2_2 = self.unpool_layers[1](output2)

        output3 = activations[-1]
        if not reshape:
            return output1, output1_2, output2, output2_2, output3
        else:
            return output1.squeeze(0), output1_2.squeeze(0), output2.squeeze(
                0), output2_2.squeeze(0), output3.squeeze(0)


class GCN(Model):

    def __init__(self, tensor_dict, args):
        super(GCN, self).__init__()
        self.tensor_dict = tensor_dict
        self.args = args
        if self.args.cnn_type == 'VGG':
            self.forward_cnn = self.forward_vgg
            self.build_cnn = self.build_vgg
        elif self.args.cnn_type == 'RES':
            self.forward_cnn = self.forward_res
            self.build_cnn = self.build_res
        elif self.args.cnn_type == 'STR':
            self.forward_cnn = self.forward_str
            self.build_cnn = self.build_str
        self.build()

    def _build(self):
        FLAGS = self.args
        self.build_cnn()
        # first project block
        self.layers.append(GraphProjection())
        self.layers.append(
            GraphConvolution(input_dim=FLAGS.feat_dim,
                             output_dim=FLAGS.hidden,
                             gcn_block_id=1,
                             tensor_dict=self.tensor_dict))

        for _ in range(12):
            self.layers.append(
                GraphConvolution(input_dim=FLAGS.hidden,
                                 output_dim=FLAGS.hidden,
                                 gcn_block_id=1,
                                 tensor_dict=self.tensor_dict))
        self.layers.append(
            GraphConvolution(input_dim=FLAGS.hidden,
                             output_dim=FLAGS.coord_dim,
                             act=None,
                             gcn_block_id=1,
                             tensor_dict=self.tensor_dict))

        # second project block
        self.layers.append(GraphProjection())
        self.layers.append(GraphPooling(tensor_dict=self.tensor_dict,
                                        pool_id=1))  # unpooling
        self.layers.append(
            GraphConvolution(input_dim=FLAGS.feat_dim + FLAGS.hidden,
                             output_dim=FLAGS.hidden,
                             gcn_block_id=2,
                             tensor_dict=self.tensor_dict))
        for _ in range(12):
            self.layers.append(
                GraphConvolution(input_dim=FLAGS.hidden,
                                 output_dim=FLAGS.hidden,
                                 gcn_block_id=2,
                                 tensor_dict=self.tensor_dict))
        self.layers.append(
            GraphConvolution(input_dim=FLAGS.hidden,
                             output_dim=FLAGS.coord_dim,
                             act=None,
                             gcn_block_id=2,
                             tensor_dict=self.tensor_dict))
        # third project block
        self.layers.append(GraphProjection())
        self.layers.append(GraphPooling(tensor_dict=self.tensor_dict,
                                        pool_id=2))  # unpooling
        self.layers.append(
            GraphConvolution(input_dim=FLAGS.feat_dim + FLAGS.hidden,
                             output_dim=FLAGS.hidden,
                             gcn_block_id=3,
                             tensor_dict=self.tensor_dict))
        for _ in range(12):
            self.layers.append(
                GraphConvolution(input_dim=FLAGS.hidden,
                                 output_dim=FLAGS.hidden,
                                 gcn_block_id=3,
                                 tensor_dict=self.tensor_dict))
        self.layers.append(
            GraphConvolution(input_dim=FLAGS.hidden,
                             output_dim=int(FLAGS.hidden / 2),
                             gcn_block_id=3,
                             tensor_dict=self.tensor_dict))
        self.layers.append(
            GraphConvolution(input_dim=int(FLAGS.hidden / 2),
                             output_dim=FLAGS.coord_dim,
                             act=None,
                             gcn_block_id=3,
                             tensor_dict=self.tensor_dict))
        self.layers = nn.ModuleList(self.layers)

        self.proj_layers = []
        for layer in self.layers:
            if layer.layer_type == 'GraphProjection':
                self.proj_layers.append(layer)

    def build_vgg(self):
        # 224 224
        self.cnn_layers_0 = []
        self.cnn_layers_0 += [
            nn.ZeroPad2d(1),
            nn.Conv2d(3, 16, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_0 += [
            nn.ZeroPad2d(1),
            nn.Conv2d(16, 16, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_0 = nn.Sequential(*self.cnn_layers_0)

        self.cnn_layers_1 = []
        self.cnn_layers_1 += [
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(16, 32, 3, 2, padding=0),
            nn.ReLU()
        ]
        # 112 112
        self.cnn_layers_1 += [
            nn.ZeroPad2d(1),
            nn.Conv2d(32, 32, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_1 += [
            nn.ZeroPad2d(1),
            nn.Conv2d(32, 32, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_1 = nn.Sequential(*self.cnn_layers_1)

        self.cnn_layers_2 = []
        self.cnn_layers_2 += [
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(32, 64, 3, 2, padding=0),
            nn.ReLU()
        ]
        # 56 56
        self.cnn_layers_2 += [
            nn.ZeroPad2d(1),
            nn.Conv2d(64, 64, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_2 += [
            nn.ZeroPad2d(1),
            nn.Conv2d(64, 64, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_2 = nn.Sequential(*self.cnn_layers_2)

        self.cnn_layers_3 = []
        self.cnn_layers_3 += [
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(64, 128, 3, 2, padding=0),
            nn.ReLU()
        ]
        # 28 28
        self.cnn_layers_3 += [
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_3 += [
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_3 = nn.Sequential(*self.cnn_layers_3)

        self.cnn_layers_4 = []
        self.cnn_layers_4 += [
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(128, 256, 5, 2, padding=0),
            nn.ReLU()
        ]
        # 14 14
        self.cnn_layers_4 += [
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_4 += [
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_4 = nn.Sequential(*self.cnn_layers_4)

        self.cnn_layers_5 = []
        self.cnn_layers_5 += [
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(256, 512, 5, 2, padding=0),
            nn.ReLU()
        ]
        # 7 7
        self.cnn_layers_5 += [
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_5 += [
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_5 += [
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_5 = nn.Sequential(*self.cnn_layers_5)

    def forward_vgg(self, img_inp):
        x = img_inp

        x = self.cnn_layers_0(x)
        x0 = x

        x = self.cnn_layers_1(x)
        x1 = x

        x = self.cnn_layers_2(x)
        x2 = x

        x = self.cnn_layers_3(x)
        x3 = x

        x = self.cnn_layers_4(x)
        x4 = x

        x = self.cnn_layers_5(x)
        x5 = x

        img_feat = [x2, x3, x4, x5]
        return img_feat

    def build_res(self):
        self.cnn_layers_01 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(3, 16, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_02 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(16, 16, 3, 1, padding=0),
            nn.ReLU()
        ]

        self.cnn_layers_11 = [
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(16, 32, 3, 2, padding=0),
            nn.ReLU()
        ]
        # 112 112
        self.cnn_layers_12 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(32, 32, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_13 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(32, 32, 3, 1, padding=0),
            nn.ReLU()
        ]

        self.cnn_layers_21 = [
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(32, 64, 3, 2, padding=0),
            nn.ReLU()
        ]
        # 56 56
        self.cnn_layers_22 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(64, 64, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_23 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(64, 64, 3, 1, padding=0),
            nn.ReLU()
        ]

        self.cnn_layers_31 = [
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(64, 128, 3, 2, padding=0),
            nn.ReLU()
        ]
        # 28 28
        self.cnn_layers_32 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_33 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.ReLU()
        ]

        self.cnn_layers_41 = [
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(128, 256, 5, 2, padding=0),
            nn.ReLU()
        ]
        # 14 14
        self.cnn_layers_42 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_43 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU()
        ]

        self.cnn_layers_51 = [
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(256, 512, 5, 2, padding=0),
            nn.ReLU()
        ]
        # 7 7
        self.cnn_layers_52 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_53 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, 3, 1, padding=0),
            nn.ReLU()
        ]
        self.cnn_layers_54 = [
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, 3, 1, padding=0),
            nn.ReLU()
        ]

        keys = list(self.__dict__.keys())[:]
        for key in keys:
            if key.startswith('cnn_layers_'):
                setattr(self, key, nn.Sequential(*getattr(self, key)))
        # print(list(self.state_dict().keys()))
    def forward_res(self, img_inp):
        x = img_inp

        x = self.cnn_layers_01(x)
        x = self.cnn_layers_02(x)
        x0 = x

        x = self.cnn_layers_11(x)
        x_res = x
        x = self.cnn_layers_12(x)
        x = self.cnn_layers_13(x) + x_res
        x1 = x

        x = self.cnn_layers_21(x)
        x_res = x
        x = self.cnn_layers_22(x)
        x = self.cnn_layers_23(x) + x_res
        x2 = x

        x = self.cnn_layers_31(x)
        x_res = x
        x = self.cnn_layers_32(x)
        x = self.cnn_layers_33(x) + x_res
        x3 = x

        x = self.cnn_layers_41(x)
        x_res = x
        x = self.cnn_layers_42(x)
        x = self.cnn_layers_43(x) + x_res
        x4 = x

        x = self.cnn_layers_51(x)
        x_res = x
        x = self.cnn_layers_52(x)
        x = self.cnn_layers_53(x) + x_res
        x = self.cnn_layers_54(x)
        x5 = x

        print('2', x2.shape)
        print('3', x3.shape)
        print('4', x4.shape)
        print('5', x5.shape)
        img_feat = [x2, x3, x4, x5]
        return img_feat

    def build_str(self):
        self.build_vgg()
        self.str_cnn_2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                       nn.ReLU())
        self.str_cnn_3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1),
                                       nn.ReLU())
        self.str_cnn_4 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1),
                                       nn.ReLU())
        self.str_cnn_5 = nn.Sequential(nn.Conv2d(1024, 512, 3, padding=1),
                                       nn.ReLU())

    def forward_str(self, img_inp, img_bis):
        features_1 = self.forward_vgg(img_inp)
        features_2 = self.forward_vgg(img_bis)

        x2 = self.str_cnn_2(torch.cat((features_1[0], features_2[0]), 1))
        x3 = self.str_cnn_3(torch.cat((features_1[1], features_2[1]), 1))
        x4 = self.str_cnn_4(torch.cat((features_1[2], features_2[2]), 1))
        x5 = self.str_cnn_5(torch.cat((features_1[3], features_2[3]), 1))
        img_feat = [x2, x3, x4, x5]
        return img_feat
