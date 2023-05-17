import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import torch_geometric
from torch_geometric import datasets
from utils import *
import json

def Make_fake_dataset(num_node, avg_degree):
    dataset = datasets.FakeDataset(avg_num_nodes = num_node, avg_degree =avg_degree)
    node_num = dataset.data.x.shape[0]
    print(node_num)
    feature = open('data/randomgraph/feature.txt', 'w')
    for i in range(node_num):
        feature.writelines(f'{i} 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1\n')
    edge_num = dataset.data.edge_index.shape[1]
    edge_list = open('data/randomgraph/edge_list.txt', 'w')
    orbit_edge = open('data/randomgraph/orbit_edge_list.txt', 'w')
    for i in range(edge_num):
        a, b  =dataset.data.edge_index[:, i]
        a, b = a.item(), b.item()
        if a< b:
            orbit_edge.write(f'{a}\t{b}\n')
        edge_list.write(f'{a} {b}\n')


def make_a_label(data, orbit):
    feature = open(f'data/{data}/feature.txt', 'r')
    node_num = len(feature.readlines())
    processing_orbit(data, node_num, orbit)
#make_a_label('randomgraph', 16)
#Make_fake_dataset(300, 22)

def Make_Last_FM_dataset():
    path = 'Last_fm_data'
    raw_edge = open(f'{path}/lastfm_asia_edges.csv', 'r')
    edge = open(f'{path}/edge.txt', 'w')
    edges = open(f'{path}/edges.txt', 'w')
    lines = raw_edge.readlines()
    for i in lines:
        a, b = i.strip().split(',')
        print(a, b)
        edges.write(f'{a} {b}\n')
        edges.write(f'{b} {a}\n')
        edge.write(f'{a}\t{b}\n')

def Make_Last_FM_features():
    path = 'Last_fm_data'
    json_file = open(f'{path}/lastfm_asia_features.json', 'rb')
    json_data = json.load(json_file)
    target = open(f'{path}/lastfm_asia_target.csv', 'r')
    target = target.readlines()
    target.pop(0)
    feature_txt = open(f'{path}/features.txt', 'w')
    print(f'Node Number: {len(json_data.keys())}')

    min_ = 10000000
    max_ = 0
    for i in range(len(json_data.items())):
        key, value  = list(json_data.items())[i]
        feature = [0 for j in range(7842)]
        for idx in value:
            feature[int(idx)] = 1
        feature =' '.join(map(str, feature))
        now_node, label = target[i].strip().split(',')
        feature = str(now_node)+ ' '+ feature + ' '+ label + '\n'
        print(len(feature))
        feature_txt.write(feature)

Make_Last_FM_features()





    #print(json.dumps(json_data, indent="\t") )


