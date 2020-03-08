import torch
import torch.optim as optim
from .losses import *
from .utils import *
from .chamfer import nn_distance_function as chamfer_dist
use_cuda = torch.cuda.is_available()


class Trainer:

    def __init__(self, tensor_dict, network, args):
        self.args = args
        self.network = network

        self.optimizer = optim.Adam(network.parameters(),
                                    lr=self.args.learning_rate)
        self.tensor_dict = tensor_dict
        self._get_loss = self._get_loss_pt

    def get_loss(self, img_inp, labels):
        if type(img_inp) != list:
            inputs = get_features(self.tensor_dict, img_inp)
            outputs = self.network(img_inp)
        else:
            inputs = get_features(self.tensor_dict, img_inp[0])
            outputs = self.network(img_inp[0].unsqueeze(0),
                                   img_inp[1].unsqueeze(0))
            outputs = [output.squeeze(0) for output in outputs]
        loss = self._get_loss(inputs, outputs, labels)
        return loss, outputs[0], outputs[2], outputs[4]

    def _get_loss_tf(self, inputs, outputs, labels):
        #output1, output1_2, output2, output2_2, output3 = outputs
        loss = 0
        loss += mesh_loss(outputs[0], labels, self.tensor_dict, 1)
        loss += mesh_loss(outputs[2], labels, self.tensor_dict, 2)
        loss += mesh_loss(outputs[4], labels, self.tensor_dict, 3)
        loss += .1 * laplace_loss(inputs, outputs[0], self.tensor_dict, 1)
        loss += laplace_loss(outputs[1], outputs[2], self.tensor_dict, 2)
        loss += laplace_loss(outputs[3], outputs[4], self.tensor_dict, 3)
        for layer in self.network.layers:
            if layer.layer_type == 'GraphConvolution':
                for key, var in layer.vars.items():
                    loss += self.args.weight_decay * torch.sum(var**2)
        return loss

    def _get_loss_pt(self, inputs, outputs, labels):
        # Edge Loss
        def edge_loss_pt(pred, labels, tensor_dict, block_id):
            gt_pts = labels[:, :3]
            idx1 = tensor_dict['edges'][block_id - 1][:, 0]
            idx2 = tensor_dict['edges'][block_id - 1][:, 0]

            if use_cuda:
                idx1 = idx1.cuda()
                idx2 = idx2.cuda()
            nod1 = torch.index_select(pred, 0, idx1)
            nod2 = torch.index_select(pred, 0, idx2)
            edge = nod1 - nod2
            # edge length loss
            edge_length = torch.sum(torch.pow(edge, 2), 1)
            edge_loss = torch.mean(edge_length) * 300
            return edge_loss

        def laplace_coord(input, tensor_dict, block_id):

            # Inputs :
            # input : nodes Tensor, size (n_pts, n_features)
            # lap_idx : laplace index matrix Tensor, size (n_pts, 10)
            #
            # Returns :
            # The laplacian coordinates of input with respect to edges as in lap_idx

            vertex = torch.cat(
                (input,
                 torch.zeros(1, 3).cuda()), 0) if use_cuda else torch.cat(
                     (input, torch.zeros(1, 3)), 0)

            indices = tensor_dict['lape_idx'][block_id - 1][:, :8].long()

            weights = tensor_dict['lape_idx'][block_id - 1][:, -1].float()

            if use_cuda:
                indices = indices.cuda()
                weights = weights.cuda()

            weights = torch.reciprocal(weights).reshape((-1, 1)).repeat((1, 3))

            num_pts, num_indices = indices.shape[0], indices.shape[1]
            indices = indices.reshape((-1,))
            vertices = torch.index_select(vertex, 0, indices)
            vertices = vertices.reshape((num_pts, num_indices, 3))

            laplace = torch.sum(vertices, 1)
            laplace = input - torch.mul(laplace, weights)

            return laplace

        def laplace_loss(input1, input2, tensor_dict, block_id):

            # Inputs :
            # input1, input2 : nodes Tensor before and after the deformation
            # lap_idx : laplace index matrix Tensor, size (n_pts, 10)
            # block_id : id of the deformation block (if different than 1 then adds
            # a move loss as in the original TF code)
            #
            # Returns :
            # The Laplacian loss, with the move loss as an additional term when relevant

            lap1 = laplace_coord(input1, tensor_dict, block_id)
            lap2 = laplace_coord(input2, tensor_dict, block_id)
            laplace_loss = torch.mean(torch.sum(torch.pow(lap1 - lap2, 2),
                                                1)) * 1500
            move_loss = torch.mean(torch.sum(torch.pow(input1 - input2, 2),
                                             1)) * 100

            if block_id == 1:
                return laplace_loss
            else:
                return laplace_loss + move_loss
            # Chamfer Loss

        pt_chamfer_loss = 0.
        pt_edge_loss = 0.
        pt_lap_loss = 0.
        lap_const = [0.2, 1., 1.]
        for idx, (output, feat) in enumerate(
                zip([outputs[0], outputs[2], outputs[4]],
                    [inputs, outputs[1], outputs[3]])):
            dist1, dist2, _, _ = chamfer_dist(output, labels[:, :3])
            pt_chamfer_loss += torch.mean(dist1) + torch.mean(dist2)
            pt_edge_loss += edge_loss_pt(output, labels, self.tensor_dict,
                                         idx + 1)
            pt_lap_loss += lap_const[idx] * laplace_loss(
                feat, output, self.tensor_dict, idx + 1)

        loss = 100 * pt_chamfer_loss + 0.1 * pt_edge_loss + 0.3 * pt_lap_loss
        return loss

    def optimizer_step(self, images, labels):
        self.optimizer.zero_grad()
        loss, output1, output2, output3 = self.get_loss(images, labels)
        loss.backward()
        self.optimizer.step()
        if not use_cuda:
            return loss.item(), output1.detach().numpy(), output2.detach(
            ).numpy(), output3.detach().numpy()
        else:
            return loss.item(), output1.detach().cpu().numpy(), output2.detach(
            ).cpu().numpy(), output3.detach().cpu().numpy()
