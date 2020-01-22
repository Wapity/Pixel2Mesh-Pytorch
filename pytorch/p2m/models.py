import torch
import torch.optim as optim
from .losses import *
from .utils import *

use_cuda = torch.cuda.is_available()


class Trainer:

    def __init__(self, tensor_dict, network, args):
        self.args = args
        self.network = network

        self.optimizer = optim.Adam(network.parameters(),
                                    lr=self.args.learning_rate)
        self.tensor_dict = tensor_dict

    def get_loss(self, img_inp, labels):
        if type(img_inp) != list:
            inputs = get_features(self.tensor_dict, img_inp)
            output1, output1_2, output2, output2_2, output3 = self.network(
                img_inp)
        else:
            inputs = get_features(self.tensor_dict, img_inp[0])
            output1, output1_2, output2, output2_2, output3 = self.network(
                img_inp[0].unsqueeze(0), img_inp[1].unsqueeze(0))
            output1, output1_2, output2, output2_2, output3 = output1.squeeze(
                0), output1_2.squeeze(0), output2.squeeze(0), output2_2.squeeze(
                    0), output3.squeeze(0)
        loss = 0
        loss += mesh_loss(output1, labels, self.tensor_dict, 1)
        loss += mesh_loss(output2, labels, self.tensor_dict, 2)
        loss += mesh_loss(output3, labels, self.tensor_dict, 3)
        loss += .1 * laplace_loss(inputs, output1, self.tensor_dict, 1)
        loss += laplace_loss(output1_2, output2, self.tensor_dict, 2)
        loss += laplace_loss(output2_2, output3, self.tensor_dict, 3)
        for layer in self.network.layers:
            if layer.layer_type == 'GraphConvolution':
                for key, var in layer.vars.items():
                    loss += self.args.weight_decay * torch.sum(var**2)
        return loss, output1, output2, output3

    def optimizer_step(self, images, labels):
        self.optimizer.zero_grad()
        loss, output1, output2, output3 = self.get_loss(images, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item(), output1.detach().numpy(), output2.detach().numpy(
        ), output3.detach().numpy()
