import torch
from chamfer import *

# def laplace_coord(pred, tensor_dict, block_id):
#     vertex = torch.cat([pred, torch.zeros((1, 3), dtype=torch.int32)], dim=0)
#     indices = tensor_dict['lape_idx'][block_id - 1][:, :8]
#     weights = tensor_dict['lape_idx'][block_id - 1][:, -1].float()
#     weights = torch.repeat_interleave(torch.reshape(torch.reciprocal(weights),
#                                                     [-1, 1]),
#                                       repeats=3,
#                                       dim=1)
#
#     laplace = torch.sum(vertex[indices], dim=1)
#     laplace = pred - torch.mul(weights, laplace)
#     return laplace


##### BY BATCH
def laplace_coord(pred, tensor_dict, block_id):
    batch_size = pred.shape[0]
    vertex = torch.cat(
        [pred, torch.zeros((batch_size, 1, 3), dtype=torch.int32)], dim=1)
    indices = tensor_dict['lape_idx'][block_id - 1][:, :8]
    weights = tensor_dict['lape_idx'][block_id - 1][:, -1].float()
    weights = torch.repeat_interleave(torch.reshape(torch.reciprocal(weights),
                                                    [-1, 1]),
                                      repeats=3,
                                      dim=1)

    laplace = torch.sum(vertex[:, indices], dim=2)
    laplace = pred - torch.mul(weights, laplace)
    return laplace
