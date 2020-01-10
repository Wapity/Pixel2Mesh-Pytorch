import torch
import torch.nn.functional as F
from .chamfer import *


def laplace_coord(pred, tensor_dict, block_id):
    vertex = torch.cat([pred, torch.zeros((1, 3))], dim=0)
    indices = tensor_dict['lape_idx'][block_id - 1][:, :8].long()
    weights = tensor_dict['lape_idx'][block_id - 1][:, -1].float()
    weights = torch.repeat_interleave(torch.reshape(torch.reciprocal(weights),
                                                    [-1, 1]),
                                      repeats=3,
                                      dim=1)

    laplace = torch.sum(vertex[indices], dim=1)
    laplace = pred - torch.mul(weights, laplace)
    return laplace


##### BY BATCH
# def laplace_coord(pred, tensor_dict, block_id):
#     batch_size = pred.shape[0]
#     vertex = torch.cat(
#         [pred, torch.zeros((batch_size, 1, 3), dtype=torch.int32)], dim=1)
#     indices = tensor_dict['lape_idx'][block_id - 1][:, :8]
#     weights = tensor_dict['lape_idx'][block_id - 1][:, -1].float()
#     weights = torch.repeat_interleave(torch.reshape(torch.reciprocal(weights),
#                                                     [-1, 1]),
#                                       repeats=3,
#                                       dim=1)
#
#     laplace = torch.sum(vertex[:, indices], dim=2)
#     laplace = pred - torch.mul(weights, laplace)
#     return laplace


def laplace_loss(pred1, pred2, tensor_dict, block_id):
    # laplace term
    lap1 = laplace_coord(pred1, tensor_dict, block_id)
    lap2 = laplace_coord(pred2, tensor_dict, block_id)
    laplace_loss = torch.mean(torch.sum(torch.pow(lap1 - lap2, 2), 1)) * 1500
    if block_id == 1:
        move_loss = torch.mean(torch.sum(torch.pow(pred1 - pred2, 2), 1)) * 100
        laplace_loss += move_loss
    return laplace_loss


def unit(tensor):
    return F.normalize(tensor, p=2, dim=1)


def mesh_loss(pred, labels, tensor_dict, block_id):
    gt_pt = labels[:, :3]  # gt points
    gt_nm = labels[:, 3:]  # gt normals

    # edge in graph
    nod1 = torch.index_select(pred, 0, tensor_dict['edges'][block_id - 1][:, 0])
    nod2 = torch.index_select(pred, 0, tensor_dict['edges'][block_id - 1][:, 1])
    edge = nod1 - nod2

    # edge length loss
    edge_length = torch.sum(torch.pow(edge, 2), dim=1)
    edge_loss = torch.mean(edge_length) * 300

    # chamer distance
    dist1, dist2, idx1, idx2 = nn_distance_function(gt_pt, pred)
    point_loss = (torch.mean(dist1) + 0.55 * torch.mean(dist2)) * 3000

    # normal cosine loss
    normal = torch.index_select(pred, 0, idx2.long())
    #normal = gt_nm[:, idx2.long()]
    normal = torch.index_select(normal, 0,
                                tensor_dict['edges'][block_id - 1][:, 0])
    #normal = normal[:, tensor_dict['edges'][block_id - 1][:, 0]]
    cosine = torch.abs(torch.sum(torch.mul(unit(normal), unit(edge)), 1))
    # cosine = tf.where(tf.greater(cosine,0.866), tf.zeros_like(cosine), cosine) # truncated
    normal_loss = torch.mean(cosine) * 0.5

    total_loss = point_loss + edge_loss + normal_loss
    return total_loss
