from utils import *
import torch
from model import *
from torch_geometric.data import DataLoader
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_curve
from torch_geometric.datasets import PPI
from numpy import argmax
import numpy as np
from torch_geometric.utils import remove_isolated_nodes

task_id = 5
train_ppi = PPI('node_dataset/ppi/', transform=get_task_rm_iso(task_id))

# edge_txt = open('edge.txt', 'w')
# edge_txt2 = open('edge2.txt', 'w')
# print(train_ppi[5].edge_index)
# self = []
# for i in range(train_ppi[5].edge_index.shape[1]):
#     a, b =  train_ppi[5].edge_index[:, i]
#     if a>b:
#         edge_txt.writelines(f'{b}\t{a}\n')
#     edge_txt2.writelines(f'{b} {a}\n')
# n = 0
feature_txt = open('features.txt' , 'w')
for i in range(train_ppi[5].x.shape[0]):
    feature = train_ppi[5].x[i,:].tolist()
    feature = list(map(str, feature))
    feature = f'{str(i)} {" ".join(feature)} {str(train_ppi[5].y[i].item())}\n'
    feature_txt.write(feature)
    print(feature)

