import torch
import numpy as np
import pickle
from skimage import io, transform
from p2m.api import GCN
from p2m.utils import *
from p2m.models import Trainer
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
args.add_argument('epochs',
                  help='Number of epochs to train.',
                  type=int,
                  default=5)
FLAGS = args
# Define tensors(dict) and model
num_blocks = 3
num_supports = 2
pkl = pickle.load(open('Data/ellipsoid/info_ellipsoid.dat', 'rb'),
                  encoding='bytes')
info_dict = construct_ellipsoid_info(pkl)
tensor_dict = {
    'features': torch.from_numpy(info_dict.features),
    'edges': [torch.from_numpy(e).long() for e in info_dict.edges],
    'faces': info_dict.faces,
    'pool_idx': info_dict.pool_idx,
    'lape_idx': [torch.from_numpy(l).float() for l in info_dict.lape_idx],
    'support1': [create_sparse_tensor(info) for info in info_dict.support1],
    'support2': [create_sparse_tensor(info) for info in info_dict.support2],
    'support3': [create_sparse_tensor(info) for info in info_dict.support3]
}

model = GCN(tensor_dict, args)
print('---- Model Created')

trainer = Trainer(tensor_dict, model, args)
print('---- Trainer Created')

img_inp = torch.randn((224, 224, 3)).permute(2, 0, 1).float()  #to change by real input
labels = torch.zeros((156, 6)) #to change by real input
trainer.optimizer_step(img_inp, labels)

train_number = 0  #data.number
for epoch in range(0 * FLAGS.epochs):
    all_loss = np.zeros(train_number, dtype='float32')
    for iters in range(train_number):
        # Fetch training data
        img_inp, y_train, data_id = data.fetch()
        img_inp, y_train, data_id = torch.randn(
            (224, 224, 3)).permute(2, 0, 1).float(), torch.zeros((156, 6)), None
        # Training step
        dists, out1, out2, out3 = trainer.optimizer_step(img_inp, labels)
        all_loss[iters] = dists
        mean_loss = np.mean(all_loss[np.where(all_loss)])
        if (iters + 1) % 128 == 0:
            print('Epoch %d, Iteration %d' % (epoch + 1, iters + 1))
            # print('Mean loss = %f, iter loss = %f, %d' %
            #       (mean_loss, dists, data.queue.qsize()))
    # Save model
    train_loss.write('Epoch %d, loss %f\n' % (epoch + 1, mean_loss))

#data.shutdown()
print('Training Finished!')
