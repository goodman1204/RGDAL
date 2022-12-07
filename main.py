# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from argparse import ArgumentParser
# from dual_gnn.cached_gcn_conv import CachedGCNConv
from dual_gnn.dataset.DomainData import DomainData
# from dual_gnn.ppmi_conv import PPMIConv
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import itertools
import model_mi as mm
from copy import deepcopy
from utils import feature_perturb, get_edge_corrupted_data,clustering_evaluation,find_best_acc,test_perturbed_target, save_results, plot_tsne,test_perturbed_tsne
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
parser = ArgumentParser()
parser.add_argument("--source", type=str, default='dblp')
parser.add_argument("--target", type=str, default='acm')
parser.add_argument("--name", type=str, default='proposed')


args = parser.parse_args()



dataset = DomainData("data/{}".format(args.source), name=args.source)
source_data = dataset[0]
print("source data:",args.source)
print(source_data)
dataset = DomainData("data/{}".format(args.target), name=args.target)
target_data = dataset[0]
print("target data:",args.target)
print(target_data)





source_data = source_data.to(device)
target_data = target_data.to(device)
# target_data_noise = target_data_noise.to(device)


# save model path

gcn_encoder_save_path = './saved_models/proposed/gcn_encoder_{}_{}_{}'.format(args.source, args.target,args.name)
cls_model_save_path = './saved_models/proposed/cls_model_{}_{}_{}'.format(args.source, args.target,args.name)


# source_best_epoch, source_best_result = find_best_acc(source_history_results)


target_data_noise = DomainData("data/{}".format(args.target), name=args.target)[0]

# perform attacks on the target data
target_data_noise = target_data_noise.to(device)
target_noise_results = test_perturbed_target(target_data_noise,gcn_encoder_save_path,cls_model_save_path)

# z, preds,true_labels= test_perturbed_tsne(target_data_noise,gcn_encoder_save_path+" "+str(best_epoch),cls_model_save_path+" "+ str(best_epoch))
# plot_tsne(args.target,args.name,target_best_epoch,z.cpu(),true_labels.cpu(),preds.cpu())

print("target_noise results:",target_noise_results)








