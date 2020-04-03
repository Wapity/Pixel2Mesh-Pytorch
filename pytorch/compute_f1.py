import torch
import numpy as np
from p2m.api import GCN
from p2m.fetcher import *
from p2m.utils import *
from p2m.external.chamfer_python import distChamfer
#from p2m.external.fscore import fscore
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
                  default='data/training_data/f1_list_stereo.txt')
args.add_argument('--cnn_type',
                  help='Type of Neural Network',
                  type=str,
                  default='STR')
args.add_argument('--checkpoint',
                  help='Checkpoint to use.',
                  type=str,
                  default='data/checkpoints/last_checkpoint_str.pt')
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

data = DataFetcher(FLAGS.f1_data,
                   stereo=FLAGS.cnn_type == 'STR',
                   compute_f1=FLAGS.num_samples)
data.setDaemon(True)
data.start()
data_number = data.number
print('---- Loadind f1 data, {} num samples'.format(data_number))


def fscore(d1, d2, threshold):
    precision_1 = torch.mean((dist1 < threshold).float())
    precision_2 = torch.mean((dist2 < threshold).float())
    f_score = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    return f_score.detach().cpu().item()


f1_tau, f2_tau = [], []
for param in model.parameters():
    param.requires_grad = False
with torch.no_grad():
    model.eval()
    all_dist_1, all_dist_2 = [], []
    for iters in range(data_number):
        torch.cuda.empty_cache()
        if FLAGS.cnn_type != 'STR':
            img_inp, y_train, data_id = data.fetch()
            img_inp, y_train = process_input(img_inp, y_train)
            if use_cuda:
                img_inp, y_train = img_inp.cuda(), y_train.cuda()
            pred_points = model(img_inp)[-1]

        else:
            img_inp_1, img_inp_2, y_train, id = data.fetch()
            img_inp_1, _ = process_input(img_inp_1, y_train)
            img_inp_2, y_train = process_input(img_inp_2, y_train)

            img_inp_1, img_inp_2 = img_inp_1.unsqueeze(0), img_inp_2.unsqueeze(
                0)
            if use_cuda:
                img_inp_1, img_inp_2, y_train = img_inp_1.cuda(
                ), img_inp_2.cuda(), y_train.cuda()
            pred_points = model(img_inp_1, img_inp_2)[-1][0]
        gt_points = y_train[:, :3]
        if use_cuda:
            dist1, dist2, _, _ = distChamfer(
                pred_points.unsqueeze(0).cuda(),
                gt_points.unsqueeze(0).cuda())
        else:
            dist1, dist2, _, _ = distChamfer(pred_points.unsqueeze(0),
                                             gt_points.unsqueeze(0))
        f1_tau.append(fscore(dist1, dist2, 0.0001))
        f2_tau.append(fscore(dist1, dist2, 0.0002))
        print('Sample = {}, f1_tau = {:.2f}, f1_2tau = {:.2f}'.format(
            iters + 1, f1_tau[-1], f2_tau[-1]))
    score_f1 = np.mean(f1_tau)
    print('------> threshold = {}, fscore = {}'.format(0.0001, score_f1))
    score_f2 = np.mean(f2_tau)
    print('------> threshold = {}, fscore = {}'.format(0.0002, score_f2))
