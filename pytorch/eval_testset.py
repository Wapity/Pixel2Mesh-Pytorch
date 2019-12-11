# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 18:55:57 2019

@author: lhuls
"""
import torch
import torch.nn as nn
import numpy as np
import pickle
import definitions
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

    
#PyTorch Config    
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

#Construct Feed dictionary
#Unpickle, latin1 encoding to deccode between python2 to python3, Can chaneg if rewrite initial encoding
pkl = pickle.load(open('Data/ellipsoid/info_ellipsoid.dat', 'rb'),encoding = 'latin1')
feed_dict = construct_feed_dict(pkl)
train_number = data.number

def f_score(label, predict, dist_label, dist_pred, threshold):
	num_label = label.shape[0]
	num_predict = predict.shape[0]

	f_scores = []
	for i in range(len(threshold)):
		num = len(np.where(dist_label <= threshold[i])[0])
		recall = 100.0 * num / num_label
		num = len(np.where(dist_pred <= threshold[i])[0])
		precision = 100.0 * num / num_predict

		f_scores.append((2*precision*recall)/(precision+recall+1e-8))
	return np.array(f_scores)
#end
    
# # Initialize session
# # xyz1:dataset_points * 3, xyz2:query_points * 3
# xyz1=tf.placeholder(tf.float32,shape=(None, 3))
# xyz2=tf.placeholder(tf.float32,shape=(None, 3))
# # chamfer distance
# dist1,idx1,dist2,idx2 = nn_distance(xyz1, xyz2)
# # earth mover distance, notice that emd_dist return the sum of all distance
# match = approx_match(xyz1, xyz2)
# emd_dist = match_cost(xyz1, xyz2, match)

# config=tf.ConfigProto()
# config.gpu_options.allow_growth=True
# config.allow_soft_placement=True
# sess = tf.Session(config=config)
# sess.run(tf.global_variables_initializer())
# model.load(sess)

###
class_name = {'02828884':'bench','03001627':'chair','03636649':'lamp','03691459':'speaker','04090263':'firearm','04379243':'table','04530566':'watercraft','02691156':'plane','02933112':'cabinet','02958343':'car','03211117':'monitor','04256520':'couch','04401088':'cellphone'}
model_number = {i:0 for i in class_name}
sum_f = {i:0 for i in class_name}
sum_cd = {i:0 for i in class_name}
sum_emd = {i:0 for i in class_name}

for iters in range(train_number):
	# Fetch training data
	img_inp, label, model_id = data.fetch()
	feed_dict.update({'img_inp': img_inp})
	feed_dict.update({'labels': label})
	# Training step
	predict = sess.run(model.output3, feed_dict=feed_dict)

	label = label[:, :3]
	d1,i1,d2,i2,emd = sess.run([dist1,idx1,dist2,idx2, emd_dist], feed_dict={xyz1:label,xyz2:predict})
	cd = np.mean(d1) + np.mean(d2)

	class_id = model_id.split('_')[0]
	model_number[class_id] += 1.0

	sum_f[class_id] += f_score(label,predict,d1,d2,[0.0001, 0.0002])
	sum_cd[class_id] += cd # cd is the mean of all distance
	sum_emd[class_id] += emd[0] # emd is the sum of all distance
	print('processed number', iters)

log = open('record_evaluation.txt', 'a')
for item in model_number:
	number = model_number[item] + 1e-8
	f = sum_f[item] / number
	cd = (sum_cd[item] / number) * 1000 #cd is the mean of all distance, cd is L2
	emd = (sum_emd[item] / number) * 0.01 #emd is the sum of all distance, emd is L1
	print(class_name[item], int(number), f, cd, emd)
	print(og, class_name[item], int(number), f, cd, emd)
log.close()
sess.close()
data.shutdown()
print('Testing Finished!')
