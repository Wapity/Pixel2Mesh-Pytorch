import torch
import numpy as np
from p2m.api import GCN
from p2m.utils import *
from p2m.fetcher import *
from p2m.models import Trainer
import argparse
from datetime import datetime
import os

# Set random seed
seed = 1024
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.empty_cache()
use_cuda = torch.cuda.is_available()

# Settings
args = argparse.ArgumentParser()

args.add_argument('--training_data',
                  help='Training data.',
                  type=str,
                  default='data/training_data/list_stereo_full.txt')
args.add_argument('--testing_data',
                  help='Testing data.',
                  type=str,
                  default='data/testing_data/test_list_str.txt')
args.add_argument('--batch_size', help='Batch size.', type=int, default=8)
args.add_argument('--learning_rate',
                  help='Learning rate.',
                  type=float,
                  default=1e-4)  # before 4.5e-5
args.add_argument('--learning_rate_decay',
                  help='Learning rate.',
                  type=float,
                  default=0.98)
args.add_argument('--learning_rate_every',
                  help='Learning rate.',
                  type=int,
                  default=2)
args.add_argument('--show_every',
                  help='Frequency of displaying loss',
                  type=int,
                  default=100)
args.add_argument('--weight_decay',
                  help='Weight decay for L2 loss.',
                  type=float,
                  default=1e-6)
args.add_argument('--epochs',
                  help='Number of epochs to train.',
                  type=int,
                  default=20)
args.add_argument('--cnn_type',
                  help='Type of Neural Network',
                  type=str,
                  default='STR')
args.add_argument(
    '--checkpoint',
    help='Checkpoint to use.',
    type=str,
    default='temp/STR/03-28_19-22-59/epoch_20/last_checkpoint.pt')  # rechanged #changed
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

mydir = os.path.join(os.getcwd(), 'temp', FLAGS.cnn_type,
                     datetime.now().strftime('%m-%d_%H-%M-%S'))
os.makedirs(mydir)
print('---- Folder created : {}'.format(mydir))

data = DataFetcher(FLAGS.training_data, stereo=True)
data.setDaemon(True)
data.start()
train_number = data.number
print('---- Loadind training data, {} num samples'.format(train_number))

test_list = []
with open(FLAGS.testing_data, 'r+', encoding="utf-8") as f:
    while (True):
        line = f.readline().strip()
        if not line:
            break
        line_1, line_2 = line.split(',')
        id = line_1.split('/')[-1][:-4]
        test_list.append((id, load_image(line_1), load_image(line_2)))
print('---- Loadind testing data, {} num samples'.format(len(test_list)))

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
print('---- Model initialized from RES')

trainer = Trainer(tensor_dict, model, FLAGS)
print('---- Trainer Created')

print('---- Training ...')
print('\n')
starter = datetime.now()
for epoch in range(FLAGS.epochs):
    if (epoch + 1) % FLAGS.learning_rate_every == 0:
        trainer.decay_lr()
    start_epoch = datetime.now()
    timer = start_epoch
    epoch_dir = mydir + '/epoch_{}'.format(epoch + 1)
    os.makedirs(epoch_dir)
    os.makedirs(epoch_dir + '/outputs')
    print('-------- Folder created : {}'.format(epoch_dir))
    all_loss = np.zeros(int(train_number / FLAGS.batch_size), dtype='float32')
    print('-------- Training epoch {} ...'.format(epoch + 1))
    for iters in range(int(train_number / FLAGS.batch_size)):
        torch.cuda.empty_cache()
        start_iter = datetime.now()
        if FLAGS.batch_size == -11:
            img_inp_1, img_inp_2, y_train, data_id = data.fetch()
            img_inp_1, _ = process_input(img_inp_1, y_train)
            img_inp_2, y_train = process_input(img_inp_2, y_train)
            if use_cuda:
                img_inp_1, img_inp_2, y_train = img_inp_1.cuda(
                ), img_inp_2.cuda(), y_train.cuda()
            img_inp = [img_inp_1, img_inp_2]
            dists, out1, out2, out3 = trainer.optimizer_step(img_inp, y_train)
            all_loss[iters] = dists
            mean_loss = np.mean(all_loss[np.where(all_loss)])
        else:
            img_inp_1, img_inp_2, y_train = [], [], []
            for _ in range(FLAGS.batch_size):
                x_1, x_2, y, id = data.fetch()
                x_1, _ = process_input(x_1, y)
                x_2, y = process_input(x_2, y)
                if use_cuda:
                    y_train.append(y.cuda())
                else:
                    y_train.append(y)
                img_inp_1.append(x_1)
                img_inp_2.append(x_2)

            img_inp_1 = torch.stack(img_inp_1)
            img_inp_2 = torch.stack(img_inp_2)
            if use_cuda:
                img_inp_1 = img_inp_1.cuda()
                img_inp_2 = img_inp_2.cuda()
            img_inp = [img_inp_1, img_inp_2]
            dists, out1, out2, out3 = trainer.optimizer_step(img_inp, y_train)
            all_loss[iters] = dists
            mean_loss = np.mean(all_loss[np.where(all_loss)])
        end_iter = datetime.now()
        if iters == 0:
            total_iter = end_iter - start_iter
            print(" REAL TIME PER IMAGE == ",
                  total_iter.seconds / FLAGS.batch_size)
        if (iters + 1) % FLAGS.show_every == 0:
            print(
                '------------ Iteration = {}, mean loss = {:.2f}, iter loss = {:.2f}'
                .format(iters + 1, mean_loss, dists))

            print("Time for iterations :", datetime.now() - timer)
            timer = datetime.now()
            print("Global time :", timer - starter)

    print('-------- Training epoch {} done !'.format(epoch + 1))
    print("Time for epoch :", timer - start_epoch)
    print("Global time :", timer - starter)

    ckp_dir = epoch_dir + '/last_checkpoint.pt'
    torch.save(model.state_dict(), ckp_dir)
    print('-------- Training checkpoint last saved !')

    print('-------- Testing epoch {} ...'.format(epoch + 1))
    for id, img_test_1, img_test_2 in test_list:
        output3 = model(img_test_1, img_test_2)[-1]
        mesh = process_output(output3)
        pred_path = epoch_dir + '/outputs/' + id + '.obj'
        np.savetxt(pred_path, mesh, fmt='%s', delimiter=' ')
    print('-------- Testing epoch {} done !'.format(epoch + 1))
    print('\n')
