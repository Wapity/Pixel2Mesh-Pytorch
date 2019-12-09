#  Copyright (C) 2019 Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, Yu-Gang Jiang, Fudan University
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from Helper.chamfer import *

def laplace_coord(pred, placeholders, block_id):
	vertex = torch.concat([pred, torch.zeros([1,3])], 0)
	indices = placeholders['lape_idx'][block_id-1][:, :8]
	weights = torch.cast(placeholders['lape_idx'][block_id-1][:,-1], torch.float32)

	weights = torch.tile(torch.reshape(torch.reciprocal(weights), [-1,1]), [1,3])
	laplace = torch.reduce_sum(torch.gather(vertex, indices), 1)
	laplace = torch.subtract(pred, torch.multiply(laplace, weights))
	return laplace

def laplace_loss(pred1, pred2, placeholders, block_id):
	# laplace term
	lap1 = laplace_coord(pred1, placeholders, block_id)
	lap2 = laplace_coord(pred2, placeholders, block_id)
	laplace_loss = torch.reduce_mean(torch.reduce_sum(torch.square(torch.subtract(lap1,lap2)), 1)) * 1500

	move_loss = torch.reduce_mean(torch.reduce_sum(torch.square(torch.subtract(pred1, pred2)), 1)) * 100
	move_loss = torch.cond(torch.equal(block_id,1), lambda:0., lambda:move_loss)
	return laplace_loss + move_loss
	
def unit(tensor):
	return torch.nn.l2_normalize(tensor, dim=1)

def mesh_loss(pred, placeholders, block_id):
	gt_pt = placeholders['labels'][:, :3] # gt points
	gt_nm = placeholders['labels'][:, 3:] # gt normals

	# edge in graph
	nod1 = torch.gather(pred, placeholders['edges'][block_id-1][:,0])
	nod2 = torch.gather(pred, placeholders['edges'][block_id-1][:,1])
	edge = torch.subtract(nod1, nod2)

	# edge length loss
	edge_length = torch.reduce_sum(torch.square(edge), 1)
	edge_loss = torch.reduce_mean(edge_length) * 300

	# chamer distance
	dist1,idx1,dist2,idx2 = nn_distance(gt_pt, pred)
	point_loss = (torch.reduce_mean(dist1) + 0.55*torch.reduce_mean(dist2)) * 3000

	# normal cosine loss
	normal = torch.gather(gt_nm, torch.squeeze(idx2, 0))
	normal = torch.gather(normal, placeholders['edges'][block_id-1][:,0])
	cosine = torch.abs(torch.reduce_sum(torch.multiply(unit(normal), unit(edge)), 1))
	# cosine = torch.where(torch.greater(cosine,0.866), torch.zeros_like(cosine), cosine) # truncated
	normal_loss = torch.reduce_mean(cosine) * 0.5

	total_loss = point_loss + edge_loss + normal_loss
	return total_loss
