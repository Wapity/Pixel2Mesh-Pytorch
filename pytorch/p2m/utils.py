import numpy as np
import torch


class AttributeDict(dict):

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def convert_dict(d):
    new_d = AttributeDict()
    new_d.update(d)
    return new_d


def create_sparse_tensor(info):
    indices = torch.LongTensor(info[0])
    values = torch.FloatTensor(info[1])
    shape = torch.Size(info[2])
    sparse_tensor = torch.sparse.FloatTensor(indices.t(), values, shape)
    return sparse_tensor


def construct_ellipsoid_info(pkl):
    """Ellipsoid info in numpy and tensor types"""
    coord = pkl[0]
    pool_idx = pkl[4]
    faces = pkl[5]
    lape_idx = pkl[7]

    edges = []
    for i in range(1, 4):
        adj = pkl[i][1]
        edges.append(adj[0])

    info_dict = {
        'features': coord,
        'edges': edges,
        'faces': faces,
        'pool_idx': pool_idx,
        'lape_idx': lape_idx,
        'support1': pkl[1],
        'support2': pkl[2],
        'support3': pkl[3]
    }
    return convert_dict(info_dict)


def get_features(tensor_dict, images):
    if len(images.shape) == 4:
        batch_size = int(images.shape[0])
        return tensor_dict['features'].data.unsqueeze(0).expand(
            batch_size, -1, -1)
    else:
        return tensor_dict['features']
