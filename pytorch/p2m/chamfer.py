#from .external.chamfer3D import dist_chamfer_3D
#nn_distance_function = chamfer3D.dist_chamfer_3D.chamfer_3DFunction
#nn_distance_module = chamfer3D.dist_chamfer_3D.chamfer_3DDist

from .external.chamfer_python import distChamfer


def nn_distance_function(a, b):
    if len(a.shape) == 3:
        return distChamfer(a, b)
    else:
        dist1, dist2, idx1, idx2 = distChamfer(a.unsqueeze(0), b.unsqueeze(0))
        return dist1.squeeze(0), dist2.squeeze(0), idx1.squeeze(
            0), idx2.squeeze(0)
