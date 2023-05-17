import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, GraphConv
import pandas as pd
import networkx as nx
from collections import deque

def load_data(data):
    data_path = f'../data/{data}/'
    edge_list = open(f'{data_path}edge_list.txt', 'r')
    edge = [[], []]
    for line in edge_list:
        node1, node2 = map(int, line.strip().split(' '))
        edge[0].append(node1)
        edge[1].append(node2)
    edge = torch.LongTensor(edge)

    feature_list = open(f'{data_path}features.txt', 'r')
    feature_list = feature_list.readlines()

    n_node = len(feature_list)
    label = torch.zeros((n_node))
    feature = torch.zeros((n_node, 50))

    for features in feature_list:
        features = list(map(float, features.strip().split()))

        node, features_, label_ = features[0], features[1:-1], features[-1]


        label[int(node)] = label_

        feature[int(node), :] = torch.FloatTensor(features_)


    return feature, label.long(), edge.long()


def result_anal(train_answer_list, train_pred_list):
    train_roc = roc_auc_score(train_answer_list, train_pred_list)
    fpr, tpr, thr = roc_curve(train_answer_list, train_pred_list)
    train_optimal_idx = np.argmax(tpr - fpr)
    train_optimal_threshold = thr[train_optimal_idx]
    train_acc = (torch.FloatTensor(train_pred_list)>train_optimal_threshold).float()

    train_acc = torch.mean((train_acc==train_answer_list).float())
    return train_acc, train_optimal_threshold