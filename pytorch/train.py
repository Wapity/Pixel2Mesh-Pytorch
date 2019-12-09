import torch
import torch.nn as nn
import numpy as np
import pickle
import definitions
from Helper.data_fetcher import Retrive_data
from Helper.fetcher import *
from Helper.utils import construct_feed_dict

# move the input and model to GPU for speed if available
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')

#Set default seed
seed = 1024
np.random.seed(seed)
torch.manual_seed(seed)


# Load data, initialize session
data = DataFetcher(definitions.data_file_path)
data.setDaemon(True) ####
data.start()

#NN instance
sess = nn.

    
#PyTorch Config    
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU


# Train graph model
train_loss = open('record_train_loss.txt', 'a')
train_loss.write('Start training, lr =  %f\n'%(definitions.learning_rate))

#Construct Feed dictionary
#Unpickle, latin1 encoding to deccode between python2 to python3, Can chaneg if rewrite initial encoding
pkl = pickle.load(open('Data/ellipsoid/info_ellipsoid.dat', 'rb'),encoding = 'latin1')
feed_dict = construct_feed_dict(pkl)


train_number = data.number
print(train_number)
for epoch in range(definitions.epochs):
    all_loss = np.zeros(train_number,dtype='float32') 
    for iters in range(train_number):
        # Fetch training data
        img_inp, y_train, data_id = data.fetch()
        feed_dict.update({'img_inp': img_inp})
        feed_dict.update({'labels': y_train})

        # Training step
        
        _, dists,out1,out2,out3 = sess.forward()
#         all_loss[iters] = dists
#         mean_loss = np.mean(all_loss[np.where(all_loss)])
#         if (iters+1) % 128 == 0:
#             print ('Epoch %d, Iteration %d'%(epoch + 1,iters + 1))
#             print ('Mean loss = %f, iter loss = %f, %d'%(mean_loss,dists,data.queue.qsize()))
#     # Save model
#     model.save(sess)
#     train_loss.write('Epoch %d, loss %f\n'%(epoch+1, mean_loss))
#     train_loss.flush()

# data.shutdown()
# print ('Training Finished!')














