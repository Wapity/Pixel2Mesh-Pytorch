import torch
import numpy as np
import pickle
from skimage import io, transform
from p2m.api import GCN
from p2m.utils import *
#import argparse

# Set random seed
seed = 1024
np.random.seed(seed)
torch.manual_seed(seed)

# Settings
#args = argparse.ArgumentParser()


class MyParser(dict):

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def add_argument(self, attr, default, help, type):
        self[attr] = default


args = MyParser()

args.add_argument('image',
                  help='Testing image.',
                  type=str,
                  default='Data/examples/square.png')
args.add_argument('learning_rate',
                  help='Initial learning rate.',
                  type=float,
                  default=0.)
args.add_argument('hidden',
                  help='Number of units in  hidden layer.',
                  type=int,
                  default=256)
args.add_argument('feat_dim',
                  help='Number of units in perceptual feature layer.',
                  type=int,
                  default=963)
args.add_argument('coord_dim',
                  help='Number of units in output layer.',
                  type=int,
                  default=3)
args.add_argument('weight_decay',
                  help='Weight decay for L2 loss.',
                  type=float,
                  default=5e-6)

FLAGS = args
# Define tensors(dict) and model
num_blocks = 3
num_supports = 2
pkl = pickle.load(open('Data/ellipsoid/info_ellipsoid.dat', 'rb'),
                  encoding='bytes')
info_dict = construct_ellipsoid_info(pkl)
tensor_dict = {
    'features': torch.from_numpy(info_dict.features),
    'edges': info_dict.edges,
    'faces': info_dict.faces,
    'pool_idx': info_dict.pool_idx,
    'lape_idx': info_dict.lape_idx,
    'support1': [create_sparse_tensor(info) for info in info_dict.support1],
    'support2': [create_sparse_tensor(info) for info in info_dict.support2],
    'support3': [create_sparse_tensor(info) for info in info_dict.support3]
}

model = GCN(tensor_dict, args)
print('---- Model Created')

model.load_state_dict(torch.load('Data/tf_vgg_checkpoint.pt'))
print('---- Model loaded from checkpoint')


def load_image(img_path):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img[np.where(img[:, :, 3] == 0)] = 255
    img = transform.resize(img, (224, 224))
    img = img[:, :, :3].astype('float32')

    return img


img_inp = load_image(FLAGS.image)
print('<--- Loaded image from : ', FLAGS.image)
img_inp = torch.from_numpy(img_inp).unsqueeze(0).permute(0, 3, 1, 2)
features = tensor_dict['features'].unsqueeze(0)

output3 = model(img_inp, features)[-1]
vert = output3.detach().numpy()[0]
vert = np.hstack((np.full([vert.shape[0], 1], 'v'), vert))
face = np.loadtxt('Data/ellipsoid/face3.obj', dtype='|S32')
mesh = np.vstack((vert, face))
pred_path = FLAGS.image.replace('.png', '.obj').replace('examples', 'outputs')
np.savetxt(pred_path, mesh, fmt='%s', delimiter=' ')

print('---> Saved mesh to     : ', pred_path)
