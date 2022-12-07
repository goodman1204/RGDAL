from collections import OrderedDict
from copy import deepcopy
import itertools
import matplotlib.pylab as plt
import numpy as np
import os.path as osp
import pickle
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import sklearn
from sklearn.manifold import TSNE
import torch
from torch.nn import Parameter, Linear
import torch.nn.functional as F
from torch.distributions.normal import Normal

import sys, os
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from torch_scatter import scatter_add
import torch_geometric.transforms as T
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, add_self_loops, softmax, degree, to_undirected
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch.autograd import Variable
from numbers import Number
# from GIB.pytorch_net.net import reparameterize, Mixture_Gaussian_reparam
# from GIB.pytorch_net.util import sample, to_cpu_recur, to_np_array, to_Variable, record_data, make_dir, remove_duplicates, update_dict, get_list_elements, to_string, filter_filename
# from GIB.util import get_reparam_num_neurons, sample_lognormal, scatter_sample, uniform_prior, compose_log, edge_index_2_csr, COLOR_LIST, LINESTYLE_LIST, process_data_for_nettack, parse_filename, add_distant_neighbors
# from GIB.DeepRobust.deeprobust.graph.targeted_attack import Nettack

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score, average_precision_score,f1_score,precision_score,recall_score

def to_np_array(*arrays, **kwargs):
		array_list = []
		for array in arrays:
				if isinstance(array, Variable):
						if array.is_cuda:
								array = array.cpu()
						array = array.data
				if isinstance(array, torch.Tensor) or isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor) or \
				   isinstance(array, torch.cuda.FloatTensor) or isinstance(array, torch.cuda.LongTensor) or isinstance(array, torch.cuda.ByteTensor):
						if array.is_cuda:
								array = array.cpu()
						array = array.numpy()
				if isinstance(array, Number):
						pass
				elif isinstance(array, list) or isinstance(array, tuple):
						array = np.array(array)
				elif array.shape == (1,):
						if "full_reduce" in kwargs and kwargs["full_reduce"] is False:
								pass
						else:
								array = array[0]
				elif array.shape == ():
						array = array.tolist()
				array_list.append(array)
		if len(array_list) == 1:
				array_list = array_list[0]
		return array_list


def get_data(
		data_type,
		train_fraction=1,
		added_edge_fraction=0,
		feature_noise_ratio=0,
		**kwargs):
		"""Get the pytorch-geometric data object.

		Args:
				data_type: Data type. Choose from "Cora", "Pubmed", "citeseer". If want the feature to be binarized, include "-bool" in data_type string.
								   if want to use largest connected components, include "-lcc" in data_type. If use random splitting with train:val:test=0.1:0.1:0.8,
								   include "-rand" in the data_type string.
				train_fraction: Fraction of training labels preserved for the training set.
				added_edge_fraction: Fraction of added (or deleted) random edges. Use positive (negative) number for randomly adding (deleting) edges.
				feature_noise_ratio: Noise ratio for the additive independent Gaussian noise on the features.

		Returns:
				A pytorch-geometric data object containing the specified dataset.
		"""
		def to_mask(idx, size):
				mask = torch.zeros(size).bool()
				mask[idx] = True
				return mask
		path = osp.join(osp.dirname(osp.realpath("__file__")), '..', 'data', data_type)
		# Obtain the mode if given:
		data_type_split = data_type.split("-")

		data_type_full = data_type
		data_type = data_type_split[0]
		mode = "lcc" if "lcc" in data_type_split else None
		boolean = True if "bool" in data_type_split else False
		split = "rand" if "rand" in data_type_split else None

		# Load data:
		info = {}
		if data_type in ["Cora", "Pubmed", "citeseer"]:
				dataset = Planetoid(path, data_type, transform=T.NormalizeFeatures())
				data = dataset[0]
				info["num_features"] = dataset.num_features
				info["num_classes"] = dataset.num_classes
				info['loss'] = 'softmax'
		else:
				raise Exception("data_type {} is not valid!".format(data_type))

		# Process the dataset according to the mode given:
		if mode is not None:
				if mode == "lcc":
						data = get_data_lcc(dataset.data)
				else:
						raise

		if boolean:
				data.x = data.x.bool().float()

		if split == "rand":
				unlabeled_share = 0.8
				val_share = 0.1
				train_share = 1 - unlabeled_share - val_share

				split_train, split_val, split_unlabeled = train_val_test_split_tabular(np.arange(data.x.shape[0]),
																																						   train_size=train_share,
																																						   val_size=val_share,
																																						   test_size=unlabeled_share,
																																						   stratify=to_np_array(data.y),
																																						   random_state=kwargs["seed"] if "seed" in kwargs else None,
																																						  )
				data.train_mask = to_mask(split_train, data.x.shape[0])
				data.val_mask = to_mask(split_val, data.x.shape[0])
				data.test_mask = to_mask(split_unlabeled, data.x.shape[0])

		# Reduce the number of training examples by randomly choosing some of the original training examples:
		if train_fraction != 1:
				try:
						train_mask_file = "../attack_data/{}/train_mask_tr_{}_seed_{}.p".format(data_type_full, train_fraction, kwargs["seed"] % 10)
						new_train_mask = pickle.load(open(train_mask_file, "rb"))
						data.train_mask = torch.BoolTensor(new_train_mask).to(data.y.device)
						print("Load train_mask at {}".format(train_mask_file))
				except:
						raise
						ids_chosen = []
						n_per_class = int(to_np_array(data.train_mask.sum()) * train_fraction / info["num_classes"])
						train_ids = torch.where(data.train_mask)[0]
						for i in range(info["num_classes"]):
								class_id_train = to_np_array(torch.where(((data.y == i) & data.train_mask))[0])
								ids_chosen = ids_chosen + np.random.choice(class_id_train, size=n_per_class, replace=False).tolist()
						new_train_mask = torch.zeros(data.train_mask.shape[0]).bool().to(data.y.device)
						new_train_mask[ids_chosen] = True
						data.train_mask = new_train_mask
						make_dir("../attack_data/{}/".format(data_type_full))
						pickle.dump(to_np_array(new_train_mask), open("../attack_data/{}/train_mask_tr_{}_seed_{}.p".format(data_type_full, train_fraction, kwargs["seed"] % 10), "wb"))

		# Add random edges for untargeted attacks:
		if added_edge_fraction > 0:
				data = add_random_edge(data, added_edge_fraction=added_edge_fraction)
		elif added_edge_fraction < 0:
				data = remove_edge_random(data, remove_edge_fraction=-added_edge_fraction)

		# Perturb features for untargeted attacks:
		if feature_noise_ratio > 0:
				x_max_mean = data.x.max(1)[0].mean()
				data.x = data.x + torch.randn(data.x.shape) * x_max_mean * feature_noise_ratio

		# For adversarial attacks:
		data.data_type = data_type
		if "attacked_nodes" in kwargs:
				attack_path = osp.join(osp.dirname(osp.realpath("__file__")), '..', 'attack_data', data_type_full)
				if not os.path.exists(attack_path):
						os.makedirs(attack_path)
				try:
						with open(os.path.join(attack_path, "test-node.pkl"), 'rb') as f:
								node_ids = pickle.load(f)
								info['node_ids'] = node_ids
								print("Load previous attacked node_ids saved in {}.".format(attack_path))
				except:
						test_ids = to_np_array(torch.where(data.test_mask)[0])
						node_ids = get_list_elements(test_ids, kwargs['attacked_nodes'])
						with open(os.path.join(attack_path, "test-node.pkl"), 'wb') as f:
								pickle.dump(node_ids, f)
						info['node_ids'] = node_ids
						print("Save attacked node_ids into {}.".format(attack_path))
		return data, info

# def feature_perturb(data,feature_noise_ratio):
		# data_c = deepcopy(data)
		# x_max_mean = data_c.x.max(1)[0].mean()
		# data_c.x = data_c.x + torch.rand(data_c.x.shape) * x_max_mean * feature_noise_ratio
		# return data_c

def feature_perturb(data,feature_noise_ratio):
		data_c = deepcopy(data)
		num_nodes = data_c.x.shape[0]
		num_nodes_perturb = int(feature_noise_ratio*num_nodes)
		idx = np.random.choice(num_nodes-1,num_nodes_perturb)
		for i in idx:
						# x_max_mean = data_c.x[i].max(1)[0].mean()
			data_c.x[i] = data_c.x[i] + torch.randn(data_c.x[i].shape)

		# data_c.x = torch.clamp(data_c.x,0,1)
		return data_c

# def feature_perturb(data,feature_noise_ratio):
		# data_c = deepcopy(data)

		# # feature_acc =  data_c.x.sum(dim=0)
		# # values, indices = feature_acc.sort()
		# # for i in indices[-1000:]:
				# # data_c.x[:,i]=0
		# data_c.x += torch.rand(data_c.x.shape)
		# return data_c

def feature_perturb_topk(data,feature_noise_ratio):
		data_c = deepcopy(data)
		x_max_mean = data_c.x.max(1)[0].mean()
		data_c.x = data_c.x + torch.randn(data_c.x.shape) * x_max_mean * feature_noise_ratio
		return data_c

def remove_edge_random(data, remove_edge_fraction):
		"""Randomly remove a certain fraction of edges."""
		data_c = deepcopy(data)
		num_edges = int(data_c.edge_index.shape[1] / 2)
		num_removed_edges = int(num_edges * remove_edge_fraction)
		edges = [tuple(ele) for ele in to_np_array(data_c.edge_index.T)]
		for i in range(num_removed_edges):
				idx = np.random.choice(len(edges))
				edge = edges[idx]
				edge_r = (edge[1], edge[0])
				edges.pop(idx)
				try:
						edges.remove(edge_r)
				except:
						pass
		data_c.edge_index = torch.LongTensor(np.array(edges).T).to(data.edge_index.device)
		return data_c


def add_random_edge(data, added_edge_fraction=0):
		"""Add random edges to the original data's edge_index."""
		if added_edge_fraction == 0:
				return data
		data_c = deepcopy(data)
		num_edges = int(data.edge_index.shape[1] / 2)
		num_added_edges = int(num_edges * added_edge_fraction)
		edges = [tuple(ele) for ele in to_np_array(data.edge_index.T)]
		added_edges = []
		for i in range(num_added_edges):
				while True:
						added_edge_cand = tuple(np.random.choice(data.x.shape[0], size=2, replace=False))
						added_edge_r_cand = (added_edge_cand[1], added_edge_cand[0])
						if added_edge_cand in edges or added_edge_cand in added_edges:
								if added_edge_cand in edges:
										assert added_edge_r_cand in edges
								if added_edge_cand in added_edges:
										assert added_edge_r_cand in added_edges
								continue
						else:
								added_edges.append(added_edge_cand)
								added_edges.append(added_edge_r_cand)
								break

		added_edge_index = torch.LongTensor(np.array(added_edges).T).to(data.edge_index.device)
		data_c.edge_index = torch.cat([data.edge_index, added_edge_index], 1)
		return data_c


def get_edge_corrupted_data(data, corrupt_fraction, is_original_included=True):
		"""Add random edges to the original data's edge_index.

		Args:
				data: PyG data instance
				corrupt_fraction: fraction of edges being removed and then the corresponding random edge added.
				is_original_included: if True, the original edges may be included in the random edges.

		Returns:
				data_edge_corrupted: new data instance where the edge is replaced by random edges.
		"""
		data_edge_corrupted = deepcopy(data)
		num_edges = int(data.edge_index.shape[1] / 2)
		num_corrupted_edges = int(num_edges * corrupt_fraction)
		edges = [tuple(item) for item in to_np_array(data.edge_index.T)]
		removed_edges = []
		num_nodes = data.x.shape[0]

		# Remove edges:
		for i in range(num_corrupted_edges):
				id = np.random.choice(range(len(edges)))
				edge = edges.pop(id)
				try:
						edge_r = edges.remove((edge[1], edge[0]))
				except:
						pass
				removed_edges.append(edge)
				removed_edges.append((edge[1], edge[0]))

		# Setting up excluded edges when adding:
		remaining_edges = list(set(edges).difference(set(removed_edges)))
		if is_original_included:
				edges_exclude = remaining_edges
		else:
				edges_exclude = edges

		# Add edges:
		added_edges = []
		for i in range(num_corrupted_edges):
				while True:
						added_edge_cand = tuple(np.random.choice(num_nodes, size=2, replace=False))
						added_edge_r_cand = (added_edge_cand[1], added_edge_cand[0])
						if added_edge_cand in edges_exclude or added_edge_cand in added_edges:
								continue
						else:
								added_edges.append(added_edge_cand)
								added_edges.append(added_edge_r_cand)
								break

		added_edge_index = torch.LongTensor(np.array(added_edges + remaining_edges).T).to(data.edge_index.device)
		data_edge_corrupted.edge_index = added_edge_index
		return data_edge_corrupted


def clustering_evaluation(labels_true, labels):
		# logger.info("------------------------clustering result-----------------------------")
		# logger.info("original dataset length:{},pred dataset length:{}".format(
				# len(labels_true), len(labels)))
		# logger.info('number of clusters in dataset: %d' % len(set(labels_true)))
		# logger.info('number of clusters estimated: %d' % len(set(labels)))
		# logger.info("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
		# logger.info("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
		# logger.info("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
		# logger.info("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
		# logger.info("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
		# logger.info("Normalized Mutual Information: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels))
		# logger.info("Purity Score: %0.3f" % purity_score(labels_true, labels))
		# logger.info("------------------------end ----------------------------------------")
		return	round(f1_score(labels_true,labels,average='macro'),4),\
						round(precision_score(labels_true,labels,average='macro'),4),\
						round(recall_score(labels_true,labels,average='macro'),4)

def find_best_acc(historical_result):

		best_acc_index = np.argmax(historical_result[:,0])
		return best_acc_index,historical_result[best_acc_index]

def test_perturbed_target(target_data_noise,gcn_encoder_path,cls_model_path,mask=None):

		gcn_encoder = torch.load(gcn_encoder_path)
		cls_model = torch.load(cls_model_path)

		encoded_output = gcn_encoder.encode(target_data_noise.x, target_data_noise.edge_index)
		logits = cls_model(encoded_output)
		# preds = logits.argmax(dim=1)
		# labels = target_data_noise.y
		preds = logits.argmax(dim=1) if mask is None else logits.argmax(dim=1)[mask]
		labels = target_data_noise.y if mask is None else target_data_noise.y[mask]
		corrects = preds.eq(labels)
		accuracy = corrects.float().mean()

		f1,pre,recall = clustering_evaluation(labels.cpu().data.numpy(),preds.cpu().data.numpy())

		return round(accuracy.item(),4),f1, pre, recall

def test_perturbed_tsne(target_data_noise,gcn_encoder_path,cls_model_path,mask=None):

		gcn_encoder = torch.load(gcn_encoder_path)
		cls_model = torch.load(cls_model_path)

		encoded_output = gcn_encoder.encode(target_data_noise.x, target_data_noise.edge_index)
		logits = cls_model(encoded_output)
		# preds = logits.argmax(dim=1)
		# labels = target_data_noise.y
		preds = logits.argmax(dim=1) if mask is None else logits.argmax(dim=1)[mask]
		labels = target_data_noise.y if mask is None else target_data_noise.y[mask]
		corrects = preds.eq(labels)
		accuracy = corrects.float().mean()

		f1,pre,recall = clustering_evaluation(labels.cpu().data.numpy(),preds.cpu().data.numpy())

		return encoded_output, preds,labels

def test_perturbed_UDAGCN_tsne(target_data_noise,gcn_encoder_path,cls_model_path,mask=None):

		gcn_encoder = torch.load(gcn_encoder_path)
		cls_model = torch.load(cls_model_path)

		encoded_output = gcn_encoder(target_data_noise.x, target_data_noise.edge_index,'target_noise')
		logits = cls_model(encoded_output)
		# preds = logits.argmax(dim=1)
		# labels = target_data_noise.y
		preds = logits.argmax(dim=1) if mask is None else logits.argmax(dim=1)[mask]
		labels = target_data_noise.y if mask is None else target_data_noise.y[mask]
		corrects = preds.eq(labels)
		accuracy = corrects.float().mean()

		f1,pre,recall = clustering_evaluation(labels.cpu().data.numpy(),preds.cpu().data.numpy())

		return encoded_output, preds,labels

def test_perturbed_target_UDAGCN(target_data_noise, gcn_encoder_path,cls_model_path,mask=None):

		gcn_encoder = torch.load(gcn_encoder_path)
		cls_model = torch.load(cls_model_path)

		encoded_output = gcn_encoder(target_data_noise.x, target_data_noise.edge_index,'target_noise')
		logits = cls_model(encoded_output)
		preds = logits.argmax(dim=1) if mask is None else logits.argmax(dim=1)[mask]
		labels = target_data_noise.y if mask is None else target_data_noise.y[mask]
		corrects = preds.eq(labels)
		accuracy = corrects.float().mean()

		f1,pre,recall = clustering_evaluation(labels.cpu().data.numpy(),preds.cpu().data.numpy())

		return round(accuracy.item(),4),f1, pre, recall

def save_results(args,normal_result,attack_result, attacked_domain = 'target'):

		direction = "{}_{}".format(args.source,args.target)
		parameters1 =f"hidden_dim:{args.hidden_dim} encoder_dim:{args.encoder_dim}\nperturb_target_edge:{args.perturb_target_edge} edge_corrupt_ratio:{args.edge_corrupt_ratio} perturb_target_feature:{args.perturb_target_feature} feature_noise_ratio:{args.feature_noise_ratio}"
		parameters2 =f"perturb_source_edge:{args.perturb_source_edge} edge_corrupt_ratio:{args.edge_corrupt_ratio} perturb_source_feature:{args.perturb_source_feature} feature_noise_ratio:{args.feature_noise_ratio}"

		fp = open('./results/{}_{}'.format(args.name,direction),'a')
		fp.write("\n{}\n{}\n".format(parameters1,parameters2))
		fp.write("normal result\n")

		for item in normal_result:
				fp.write("& {}	".format(item))
		fp.write("\n")

		if args.perturb_target_edge and not args.perturb_target_feature:
			fp.write("target structure attack result\n")
		elif args.perturb_target_feature and not args.perturb_target_edge:
			fp.write("target feature attack result\n")
		elif args.perturb_target_edge and args.perturb_target_feature:
			fp.write("target feature and structure attack result\n")
		elif args.perturb_source_edge and not args.perturb_source_feature:
			fp.write("source structure attack result\n")
		elif  args.perturb_source_feature and not args.perturb_source_edge:
			fp.write("source feature attack result\n")
		elif  args.perturb_source_feature and args.perturb_source_edge:
			fp.write("source feature and structure attack result\n")
		else:
			fp.write("no attack result\n")
		for item in attack_result:
			fp.write("& {} ".format(item))
		fp.write("\n")

		fp = open('./results/{}_{}_source_edge{}_source_feature{}_target_edge{}_target_feature{}_edge_p{}_feature_n{}_parameters'.format(args.name,direction,args.perturb_source_edge,args.perturb_source_feature,args.perturb_target_edge,args.perturb_target_feature,args.edge_corrupt_ratio,args.feature_noise_ratio),'w')
		for item in attack_result:
			fp.write("{} ".format(item))
		fp.write("\n")

def getSoftAssignments(latent_space, cluster_centers, num_samples):
	'''
	Returns cluster membership distribution for each sample
	:param latent_space: latent space representation of inputs
	:param cluster_centers: the coordinates of cluster centers in latent space
	:param num_clusters: total number of clusters
	:param latent_space_dim: dimensionality of latent space
	:param num_samples: total number of input samples
	:return: soft assigment based on the equation qij = (1+|zi - uj|^2)^(-1)/sum_j'((1+|zi - uj'|^2)^(-1))
	'''
	# z_expanded = latent_space.reshape((num_samples, 1, latent_space_dim))
	# z_expanded = T.tile(z_expanded, (1, num_clusters, 1))
	# u_expanded = T.tile(cluster_centers, (num_samples, 1, 1))

	# distances_from_cluster_centers = (z_expanded - u_expanded).norm(2, axis=2)
	# qij_numerator = 1 + distances_from_cluster_centers * distances_from_cluster_centers
	# qij_numerator = 1 / qij_numerator
	# normalizer_q = qij_numerator.sum(axis=1).reshape((num_samples, 1))

	# return qij_numerator / normalizer_q


	distances_from_cluster_centers = (latent_space.unsqueeze(1)- cluster_centers.unsqueeze(0)).norm(2, dim=2)
	qij_numerator = 1 + distances_from_cluster_centers * distances_from_cluster_centers
	qij_numerator = 1 / qij_numerator
	normalizer_q = qij_numerator.sum(dim=1).reshape((num_samples, 1))

	return qij_numerator / normalizer_q

def getKLDivLossExpression(Q_expression, P_expression):
	# Loss = KL Divergence between the two distributions
	log_arg = P_expression / Q_expression
	log_exp = torch.log(log_arg)
	sum_arg = P_expression * log_exp
	loss = torch.sum(sum_arg)
	return loss

def calculateP(Q):
	# Function to calculate the desired distribution Q^2, for more details refer to DEC paper
	f = Q.sum(dim=0)
	pij_numerator = Q * Q
	# pij_numerator = Q
	pij_numerator = pij_numerator / f
	normalizer_p = pij_numerator.sum(dim=1).reshape((Q.shape[0], 1))
	P = pij_numerator / normalizer_p
	return P

def plot_tsne(dataset,model_name,epoch,z,true_label,pred_label):

	tsne = TSNE(n_components=2, init='pca')
	data = z.detach().numpy()
	zs_tsne = tsne.fit_transform(data)

	# print('zs_tnse',zs_tsne.shape)


	true_label = true_label.tolist()
	# print('true_labels:',true_label)
	cluster_labels=set(true_label)
	# print(cluster_labels)
	index_group= [np.array(true_label)==y for y in cluster_labels]
	# print(index_group)
	colors = cm.tab20(range(len(index_group)))

	fig, ax = plt.subplots(figsize=[5,5])
	for index,c in zip(index_group,colors):
		ax.scatter(zs_tsne[np.ix_(index), 0], zs_tsne[np.ix_(index), 1],color=c,s=20,alpha=0.8)
	ax.axis('off')
	# ax.legend(cluster_labels)

	# ax.scatter(zs_tsne[z.shape[0]:, 0], zs_tsne[z.shape[0]:, 1],marker='^',color='b',s=40)
	# plt.title('true label')
	# ax.legend()
	plt.tight_layout()
	plt.savefig("./visualization/{}_{}_{}_tsne_{}.pdf".format(model_name,dataset,epoch,'true_label'))


	pred_label = pred_label.tolist()
	cluster_labels=set(pred_label)
	index_group= [np.array(pred_label)==y for y in cluster_labels]
	colors = cm.tab20(range(len(index_group)))
	# print('colors shape',colors.shape)

	fig, ax = plt.subplots(figsize=[5,5])
	for index,c in zip(index_group,colors):
		ax.scatter(zs_tsne[np.ix_(index), 0], zs_tsne[np.ix_(index), 1],color=c,s=20,alpha=0.8)

	# for index,c in enumerate(colors):
		# ax.scatter(zs_tsne[z.shape[0]+index:z.shape[0]+index+1, 0], zs_tsne[z.shape[0]+index:z.shape[0]+index+1, 1],marker='^',color=c,s=40)

	# ax.legend(cluster_labels)
	ax.axis('off')
	# plt.title('pred label')
	# ax.legend()
	plt.tight_layout()
	plt.savefig("./visualization/{}_{}_{}_tsne_{}.pdf".format(model_name,dataset,epoch,'pred_label'))
