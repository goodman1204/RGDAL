# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from argparse import ArgumentParser
# from dual_gnn.cached_gcn_conv import CachedGCNConv
from dual_gnn.dataset.DomainData import DomainData
from dual_gnn.ppmi_conv import PPMIConv
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import itertools
import model_mi as mm
from copy import deepcopy
from utils import feature_perturb, get_edge_corrupted_data,clustering_evaluation,find_best_acc,test_perturbed_target, save_results
import utils
from tqdm import tqdm

from torch_geometric.nn import GCNConv,VGAE, GATConv, SAGEConv, GINConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='citationv1')
parser.add_argument("--target", type=str, default='acmv9')
parser.add_argument("--name", type=str, default='proposed')
parser.add_argument("--seed", type=int,default=200)
parser.add_argument("--epochs", type=int,default=100)
parser.add_argument("--UDAGCN", type=bool,default=True)
parser.add_argument("--encoder_dim", type=int, default=128)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--mi_constraint", type=float, default=200.0)
parser.add_argument("--beta_t", type=float, default=0.0001)
parser.add_argument("--beta_s", type=float, default=0.0001)
parser.add_argument("--alpha", type=float, default=0.00001)
parser.add_argument("--kl", type=int, default=0)
parser.add_argument("--perturb_source_feature", type=int, default=0)
parser.add_argument("--perturb_target_feature", type=int, default=0)
parser.add_argument("--feature_noise_ratio", type=float, default=0.01)
parser.add_argument("--perturb_source_edge", type=int, default=0)
parser.add_argument("--perturb_target_edge", type=int, default=0)
parser.add_argument("--edge_corrupt_ratio", type=float, default=0.05)


args = parser.parse_args()
options = vars(args)
seed = args.seed
use_UDAGCN = args.UDAGCN
encoder_dim = args.encoder_dim



id_command = "source: {}, target: {}, seed: {}, UDAGCN: {}, encoder_dim: {}"\
		.format(args.source, args.target, seed, use_UDAGCN,  encoder_dim)

print(id_command)


rate = 0.0
seed = random.randint(0,205)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

dataset = DomainData("data/{}".format(args.source), name=args.source)
source_data = dataset[0]

random_node_indices = np.random.permutation(source_data.y.shape[0])
training_size = int(len(random_node_indices)*0.5)
val_size = int(len(random_node_indices) * 0.1)
train_node_indices = random_node_indices[:training_size]
val_node_indices = random_node_indices[training_size:training_size + val_size]
test_node_indices = random_node_indices[training_size + val_size:]

train_masks = torch.zeros([source_data.y.shape[0]], dtype=torch.uint8)
train_masks[train_node_indices] = 1
source_data.train_mask = train_masks

val_masks = torch.zeros([source_data.y.shape[0]], dtype=torch.uint8)
val_masks[val_node_indices] = 1

test_masks = torch.zeros([source_data.y.shape[0]], dtype=torch.uint8)
test_masks[test_node_indices] = 1
source_data.test_mask = test_masks

print(source_data)
dataset = DomainData("data/{}".format(args.target), name=args.target)
target_data = dataset[0]
print(target_data)


if args.perturb_source_feature:
	print('perturb source feature')
	source_data = feature_perturb(source_data,args.feature_noise_ratio)

if args.perturb_source_edge:
	print('perturb source edge')
	source_data = get_edge_corrupted_data(source_data, args.edge_corrupt_ratio)

if not args.perturb_source_edge and not args.perturb_source_feature:
	print("no perturb on source data")
	# target_data_noise = target_data



source_data = source_data.to(device)
target_data = target_data.to(device)
# target_data_noise = target_data_noise.to(device)
num_classes = dataset.num_classes


class GradReverse(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x):
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		grad_output = grad_output.neg()
		return grad_output, None


class GRL(nn.Module):
	def forward(self, input):
		return GradReverse.apply(input)


loss_func = nn.CrossEntropyLoss().to(device)

encoder = mm.Stochastic_encoder1(dataset.num_features,args.hidden_dim,args.encoder_dim)

print("encoder structures: ", encoder)
gcn_encoder = VGAE(encoder).to(device)
print("gcn structures: ", gcn_encoder)



# encoder = GNN(type="gcn").to(device)
# if use_UDAGCN:
	# ppmi_encoder = GNN(base_model=encoder, type="ppmi", path_len=10).to(device)


cls_model = nn.Sequential(
		nn.Linear(encoder_dim, dataset.num_classes),
		).to(device)

domain_model = nn.Sequential(
		GRL(),
		nn.Linear(encoder_dim, 40),
		nn.Dropout(),
		nn.Linear(40, 2),
		).to(device)


class Attention(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		self.dense_weight = nn.Linear(in_channels, 1)
		self.dropout = nn.Dropout(0.1)


	def forward(self, inputs):
		stacked = torch.stack(inputs, dim=1)
		weights = F.softmax(self.dense_weight(stacked), dim=1)
		outputs = torch.sum(stacked * weights, dim=1)
		return outputs


att_model = Attention(encoder_dim).to(device)

# models = [encoder, cls_model, domain_model]
models = [gcn_encoder, cls_model, domain_model]
# if use_UDAGCN:
	# models.extend([ppmi_encoder, att_model])
params = itertools.chain(*[model.parameters() for model in models])
optimizer = torch.optim.Adam(params, weight_decay=5e-4,lr=2e-3)

decayRate = 1
my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=decayRate)

def encode(data, mask=None):
	encoded_output = gcn_encoder.encode(data.x, data.edge_index)
	if mask is not None:
		encoded_output = encoded_output[mask]
	return encoded_output


# def ppmi_encode(data, cache_name, mask=None):
	# encoded_output = ppmi_encoder(data.x, data.edge_index, cache_name)
	# if mask is not None:
		# encoded_output = encoded_output[mask]
	# return encoded_output


# def encode(data, cache_name, mask=None):
	# gcn_output = encode(data, cache_name, mask)
	# if use_UDAGCN:
		# ppmi_output = ppmi_encode(data, cache_name, mask)
		# outputs = att_model([gcn_output, ppmi_output])
		# return outputs
	# else:
		# return gcn_output

def predict(data,  mask=None):
	encoded_output = encode(data, mask)
	logits = cls_model(encoded_output)
	return logits


def evaluate(preds, labels):
	corrects = preds.eq(labels)
	accuracy = corrects.float().mean()
	return accuracy


def test(data, mask=None):
	for model in models:
		model.eval()
	logits = predict(data, mask)
	preds = logits.argmax(dim=1)
	labels = data.y if mask is None else data.y[mask]
	accuracy = evaluate(preds, labels)
	f1,pre,recall = clustering_evaluation(labels.cpu().data.numpy(),preds.cpu().data.numpy())
	return round(accuracy.item(),4),f1, pre, recall


epochs = args.epochs
def train(epoch):
	for model in models:
		model.train()

	global rate
	rate = min((epoch + 1) / epochs, 0.05)

	encoded_source = gcn_encoder.encode(source_data.x,source_data.edge_index)
	kl_loss_s= gcn_encoder.kl_loss()
	recon_loss_s = gcn_encoder.recon_loss(encoded_source,source_data.edge_index)
	source_loss = kl_loss_s + recon_loss_s

	encoded_target = gcn_encoder.encode(target_data.x,target_data.edge_index)
	kl_loss_t= gcn_encoder.kl_loss()
	recon_loss_t = gcn_encoder.recon_loss(encoded_target,target_data.edge_index)
	target_loss= recon_loss_t +0.01*kl_loss_t
	# target_loss= (1 / target_data.x.size()[0]) * kl_loss_t + recon_loss_t

	source_logits = cls_model(encoded_source)

	# use source classifier loss:
	cls_loss = loss_func(source_logits[source_data.train_mask], source_data.y[source_data.train_mask])

	# for model in models:
		# for name, param in model.named_parameters():
			# if "weight" in name:
				# cls_loss = cls_loss + param.mean() * 3e-3

	# self clustering loss
	# cluster_centers = []

	# for label in range(num_classes):
		# center =  torch.mean(encoded_source[source_data.y==label],dim=0)
		# cluster_centers.append(center)
	# centers = torch.stack(cluster_centers)
	# assert centers.shape == (num_classes,encoded_source.shape[1])

	# # print("centers\n",centers.shape)

	# Q = utils.getSoftAssignments(encoded_target,centers,encoded_target.shape[0])
	# # print("Q shape:",Q.shape)
	# assert Q.shape == (encoded_target.shape[0],num_classes)

	# values, indices = torch.max(Q,dim=1)
	# print(values)
	# flag = values>0.8
	# print("sum of flag:",sum(flag))

	# target_logits = cls_model(encoded_target[flag])
	# cls_loss_target = loss_func(target_logits,indices[flag])

	# P = utils.calculateP(Q)
	# kl_loss_clustering = utils.getKLDivLossExpression(Q,P)







	# if use_UDAGCN:
	if 1:
		# use domain classifier loss:
		source_domain_preds = domain_model(encoded_source)
		target_domain_preds = domain_model(encoded_target)

		source_domain_cls_loss = loss_func(
				source_domain_preds,
				torch.ones(source_domain_preds.size(0)).type(torch.LongTensor).to(device)
				)
		target_domain_cls_loss = loss_func(
				target_domain_preds,
				torch.zeros(target_domain_preds.size(0)).type(torch.LongTensor).to(device)
				)
		loss_grl = source_domain_cls_loss + target_domain_cls_loss
		loss = cls_loss + target_loss
		print("cls_loss:{},target_loss:{}".format(cls_loss,target_loss))

		# use target classifier loss:
		# target_logits = cls_model(encoded_target)
		# target_probs = F.softmax(target_logits, dim=-1)
		# target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)

		# loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

		# loss = loss + loss_entropy * (epoch / epochs * 0.01)

		# loss = loss + args.beta_t*(mi_target-args.mi_constraint) + args.beta_s*(mi_source -  args.mi_constraint)

		# loss = loss + (mi_target + mi_source)

	else:
		loss = cls_loss + target_loss + loss_grl
		# loss = cls_loss +source_loss + target_loss

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	my_lr_scheduler.step()
	print(f"learning rate:{my_lr_scheduler.get_last_lr()}")

	# if args.kl == 1:
		# return [loss.item(),cls_loss.item(),loss_entropy.item(),source_domain_cls_loss.item(),target_domain_cls_loss.item(),mi_source_[-1],mi_target_[-1]]
	# else:
		# return [loss.item(),cls_loss.item(),loss_entropy.item(),source_domain_cls_loss.item(),target_domain_cls_loss.item(),0,0]

best_source_acc = 0.0
best_target_acc = 0.0
best_target_noise_acc = 0.0
best_epoch = 0.0
best_noise_epoch = 0.0
loss_list=[]
source_acc_list = []
target_acc_list = []
source_history_results=[]
target_history_results=[]

# save model path

gcn_encoder_save_path = './saved_models/gcn_encoder_{}_{}_{}'.format(args.source, args.target,args.name)
cls_model_save_path = './saved_models/cls_model_{}_{}_{}'.format(args.source, args.target,args.name)

for epoch in range(0, epochs):
	loss_temp = train(epoch)
	loss_list.append(loss_temp)
	source_results = test(source_data, source_data.test_mask)
	target_results = test(target_data)
	source_history_results.append(source_results)
	target_history_results.append(target_results)
	source_correct= source_results[0]
	target_correct = target_results[0]
	# target_noise_correct = test(target_data_noise)
	source_acc_list.append(source_results[0])
	target_acc_list.append(target_results[0])
	print("Epoch: {}, source: {}, target: {}".format(epoch,source_results,target_results))
	if target_correct > best_target_acc:
		best_target_acc = target_correct
		best_source_acc = source_correct
		best_epoch = epoch

		torch.save(gcn_encoder, gcn_encoder_save_path+ " "+ str(best_epoch))
		torch.save(cls_model, cls_model_save_path+ " "+ str(best_epoch))

	# if target_noise_correct > best_target_noise_acc:
		# best_target_noise_acc = target_noise_correct
		# best_noise_epoch = epoch
print("=============================================================")

# source_best_epoch, source_best_result = find_best_acc(source_history_results)
target_best_epoch, target_best_result = find_best_acc(np.array(target_history_results))


target_data_noise = DomainData("data/{}".format(args.target), name=args.target)[0]

print(f'target_data id:{id(target_data)}')
print(f'target_data_noise id:{id(target_data_noise)}')

mask = None
if args.perturb_target_feature:
	print('perturb target feature')
	target_data_noise= feature_perturb(target_data_noise,args.feature_noise_ratio)
	print(f'target_data_noise id:{id(target_data_noise)}')

if args.perturb_target_edge:
	print('perturb target edge')
	target_data_noise = get_edge_corrupted_data(target_data_noise, args.edge_corrupt_ratio)
	print(f'target_data_noise id:{id(target_data_noise)}')

if not args.perturb_target_edge and not args.perturb_target_feature:
	print("no perturb")
	# target_data_noise = target_data

# perform attacks on the target data

line = "Epoch: {}, best_source: {}, best_target: {}".format(target_best_epoch, source_history_results[target_best_epoch],target_best_result)
print(options)
print(line)
# print("target_noise results:",target_noise_results)

# save results to file





