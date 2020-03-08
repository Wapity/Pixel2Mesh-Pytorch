#from .external.chamfer3D import dist_chamfer_3D
#nn_distance_function = chamfer3D.dist_chamfer_3D.chamfer_3DFunction
#nn_distance_module = chamfer3D.dist_chamfer_3D.chamfer_3DDist
import torch
from .external.chamfer_python import distChamfer
from .external.chamfer3D.dist_chamfer_3D import chamfer_3DFunction
use_cuda = torch.cuda.is_available()
if use_cuda:
    dist_module = chamfer_3DFunction()
else:
    dist_module = distChamfer


def nn_distance_function(a, b):
    if len(a.shape) == 3:
        return dist_module(a, b)
    else:
        dist1, dist2, idx1, idx2 = dist_module(a.unsqueeze(0), b.unsqueeze(0))
        return dist1.squeeze(0), dist2.squeeze(0), idx1.squeeze(
            0), idx2.squeeze(0)
