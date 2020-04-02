import torch
import numpy as np
from p2m.api import GCN
from p2m.fetcher import *
from p2m.utils import *
from p2m.external.chamfer_python import distChamfer
from p2m.external.fscore import fscore
import argparse
from datetime import datetime
use_cuda = torch.cuda.is_available()
# Set random seed
seed = 1024
np.random.seed(seed)
torch.manual_seed(seed)

# Settings
args = argparse.ArgumentParser()
args.add_argument('--num_samples', help='num samples', type=int, default=1000)
args.add_argument('--f1_data',
                  help='F1 score data.',
                  type=str,
                  default='data/training_data/trainer_res.txt')
args.add_argument('--cnn_type',
                  help='Type of Neural Network',
                  type=str,
                  default='RES')
args.add_argument('--checkpoint',
                  help='Checkpoint to use.',
                  type=str,
                  default='data/checkpoints/last_checkpoint_res.pt')
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

if use_cuda:
    model.load_state_dict(torch.load(FLAGS.checkpoint), strict=False)
    model = model.cuda()
else:
    model.load_state_dict(torch.load(FLAGS.checkpoint,
                                     map_location=torch.device('cpu')),
                          strict=False)
print('---- Model loaded from checkpoint')

data = DataFetcher(FLAGS.f1_data, compute_f1=FLAGS.num_samples)
data.setDaemon(True)
data.start()
data_number = data.number
print('---- Loadind f1 data, {} num samples'.format(data_number))

all_dist_1, all_dist_2 = [], []
for iters in range(data_number):
    print(iters)
    img_inp, y_train, data_id = data.fetch()
    img_inp, y_train = process_input(img_inp, y_train)
    gt_points = y_train[:, :3]
    if use_cuda:
        img_inp, y_train = img_inp.cuda(), y_train.cuda()
    pred_points = model(img_inp)[-1]
    dist1, dist2, _, _ = distChamfer(
        pred_points.unsqueeze(0).cuda(),
        gt_points.unsqueeze(0).cuda())
    all_dist_1.append(dist1.squeeze(0))
    all_dist_2.append(dist2.squeeze(0))
dist1 = torch.stack(all_dist_1)
dist2 = torch.stack(all_dist_2)

threshold = 0.0001
score_f1 = fscore(dist1, dist2, threshold)[0].mean().detach().cpu().item()
print('------> threshold = {}, fscore = {}'.format(threshold, score_f1))
threshold = 0.0002
score_f1 = fscore(dist1, dist2, threshold)[0].mean().detach().cpu().item()
print('------> threshold = {}, fscore = {}'.format(threshold, score_f1))
