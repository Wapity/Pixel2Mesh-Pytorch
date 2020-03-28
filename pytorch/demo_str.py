import torch
import numpy as np
from p2m.api import GCN
from p2m.utils import *
import argparse
from datetime import datetime
use_cuda = torch.cuda.is_available()

# Set random seed
seed = 1024
np.random.seed(seed)
torch.manual_seed(seed)

# Settings
args = argparse.ArgumentParser()

args.add_argument('--image',
                  help='Testing image.',
                  type=str,
                  default='data/examples/square.png')
args.add_argument('--image_bis',
                  help='Testing image.',
                  type=str,
                  default='data/examples/square.png')
args.add_argument('--cnn_type',
                  help='Type of Neural Network',
                  type=str,
                  default='STR')
args.add_argument('--checkpoint',
                  help='Checkpoint to use.',
                  type=str,
                  default='data/checkpoints/last_checkpoint.pt')  # change with str
args.add_argument('--info_ellipsoid',
                  help='Initial Ellipsoid info',
                  type=str,
                  default='data/ellipsoid/info_ellipsoid.dat')
args.add_argument('--hidden',
                  help='Number of units in  hidden layer.',
                  type=int,
                  default=256)
args.add_argument('--feat_dim',
                  help='Number of units in perceptual feature layer.',
                  type=int,
                  default=963)
args.add_argument('--coord_dim',
                  help='Number of units in output layer.',
                  type=int,
                  default=3)

FLAGS = args.parse_args()

tensor_dict = construct_ellipsoid_info(FLAGS)
print('---- Build initial ellispoid info')

model = GCN(tensor_dict, FLAGS)
print('---- Model Created')

model.load_state_dict(
    torch.load(FLAGS.checkpoint, map_location=torch.device('cpu')))
print('---- Model loaded from checkpoint')

img_inp_1 = load_image(FLAGS.image)
print('<--- Loaded image 1 from : ', FLAGS.image)

img_inp_2 = load_image(FLAGS.image_bis)
print('<--- Loaded image 2 from : ', FLAGS.image)

output3 = model(img_inp_1, img_inp_2)[-1]
print('---- Model applied to image')

mesh = process_output(output3)
print('---- Mesh created applied to image')

pred_path = FLAGS.image.replace('.png', '.obj').replace(
    'examples/',
    'outputs/str_{}_'.format(datetime.now().strftime('%m-%d_%H-%M')))
np.savetxt(pred_path, mesh, fmt='%s', delimiter=' ')
print('---> Saved mesh to     : ', pred_path)
