import torch
import sys
import os
sys.path.append(os.pardir)
from model import *
import networkx as nx
from collections import deque
import matplotlib.pyplot as plt


def load_model(data, model_name,agg):
    data_path = f'../data/{data}/'
    if agg:
        model_path = f'../data/{data}/agg_model/{model_name}'
    else:
        model_path = f'../data/{data}/model/{model_name}'

    model_dict = torch.load(model_path)

    if model_name.startswith('ppi'):

        #hidden_state = list(model_dict.values())[1].shape[0]

        #input_dim = list(model_dict.values())[0].shape[1]

        model = Net(50, 2, 100)
        model.load_state_dict(model_dict)

        return model

    else:
        hidden_state = list(model_dict.values())[0].shape[0]
        label = list(model_dict.values())[-2].shape[0]
        input_dim = list(model_dict.values())[1].shape[1]
        if agg:
            model = eval(f'agg_GCN{3}({input_dim}, {label}, {hidden_state})')
        else:
            model = eval(f'GCN{3}({input_dim}, {label}, {hidden_state})')

    model.load_state_dict(model_dict)

    return model

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
    feature_dim = len(feature_list[0].strip().split(' ')) - 2
    n_node = len(feature_list)
    label = torch.zeros((n_node))
    feature = torch.zeros((n_node, feature_dim ))

    for features in feature_list:
        features = list(map(float, features.strip().split()))
        node, features_, label_ = features[0], features[1:-1], features[-1]

        label[int(node)] = label_
        feature[int(node), :] = torch.FloatTensor(features_)


    return feature, label.long(), edge.long()

def make_graph(data):
    graph = nx.Graph()
    edge_list = open(f'../data/{data}/edge_list.txt', 'r')
    for i in edge_list:
        node1, node2 = map(int, i.strip().split(' '))
        graph.add_edge(node1, node2)
    return graph

def plot_nhop(graph, node= 0, hop = 3,label=None):
    hop_graph = nx.Graph()
    colerdict = {0:'black', 1:'blue', 2:'orange',3:'red',4:'green', 5:'yellow', 6:'purple', 7:'grey'}
    neighbor_nodes = set([node])
    for i in range(hop):
        hop_neighbor = set()
        for j in neighbor_nodes:
            hop_neighbor = hop_neighbor|set(graph.neighbors(j))
        neighbor_nodes = neighbor_nodes|hop_neighbor
    neighbor_nodes = list(neighbor_nodes)
    len_neighbor = len(neighbor_nodes)
    for i in range(0, len_neighbor-1):
        for j in range(i, len_neighbor):
            node1 = neighbor_nodes[i]
            node2 = neighbor_nodes[j]
            if graph.has_edge(node1, node2):
                hop_graph.add_edge(node1, node2)

    color_map = [colerdict[label[i].item()] for i in hop_graph.nodes]

    nx.draw(hop_graph, pos=nx.kamada_kawai_layout(hop_graph), node_color = color_map, with_labels=True, node_size=100)
    plt.show()

def find_baco_gt(graph, node, label):
    if label[node] == 0 or label[node] == 4:
        return
    if label[node] in [5, 7, 6]:
        ground_dict = {5:0, 6:0, 7:1}
    else:
        ground_dict = {1:0, 2:0, 3:1}
    neighbor_nodes = deque([node])
    ground_nodes = []
    while True:
        now_node = neighbor_nodes.popleft()

        now_label = label[now_node].item()

        if now_label in ground_dict.keys():
            if ground_dict[now_label] < 2:
                ground_nodes.append(now_node)
                ground_dict[now_label] += 1
                neighbor_nodes += graph.neighbors(now_node)
            if sum(ground_dict.values()) == 6:
                return ground_nodes

def gnnexplainer_gnn(node_idx, gnnexplainer, x, edge_index, data):
    result = gnnexplainer.explain_node(node_idx, x, edge_index)
    important_order = torch.argsort(result[1], descending=True)

    if data == 'bashapes':
        return edge_index[:,important_order[:12]]
    elif data == 'bac':
        return edge_index[:, important_order[:12]]
    else:
        return edge_index[:, important_order[:12]]
#
#
# def fidelity(model, target_node, feature, edge, explaine_edge, label, ppi=None):
#     with torch.no_grad():
#         answer_predict = model(feature, edge).detach()
#         if ppi:
#             answer_predict = torch.sigmoid(answer_predict)[target_node, :]
#
#         else:
#             answer_predict = torch.softmax(answer_predict, dim=1)[target_node, label[target_node]]
#
#         tuple_edge= []
#         edge_index = [i for i in range(edge.shape[1])]
#
#         edge_weights = torch.ones(edge.size(1))
#         for i in range(edge.shape[1]):
#             a, b = edge[:,i]
#             tuple_edge.append((a.item(), b.item()))
#
#         for i in range(explaine_edge.shape[1]):
#             i_edge =  tuple_edge.index((explaine_edge[0,i].item(),explaine_edge[1,i].item()))
#             edge_weights[i_edge] = 0
#             edge_index.remove(i_edge)
#
#         device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
#         edge_weights = edge_weights.to(device)
#         processing_predict = model(feature, edge,  edge_weights = edge_weights)
#         if ppi:
#             processing_predict = torch.sigmoid(processing_predict)[target_node, :]
#         else:
#             processing_predict = torch.softmax(processing_predict, dim=1)[target_node, label[target_node]].detach()
#
#
#     return answer_predict - processing_predict

def fidelity(model, target_node, feature, edge, explaine_edge, label, ppi=None):
    with torch.no_grad():
        device = torch.device(edge.device)



        label = label.to(device)

        answer_predict = model(feature, edge).detach()

        if ppi:
            answer_predict = torch.sigmoid(answer_predict)[target_node, :]

        else:
            answer_predict = torch.softmax(answer_predict, dim=1)[target_node, label[target_node]]

        tuple_edge= []
        edge_index = [i for i in range(edge.shape[1])]

        edge_weights = torch.ones(edge.size(1))
        for i in range(edge.shape[1]):
            a, b = edge[:,i]
            tuple_edge.append((a.item(), b.item()))

        for i in range(explaine_edge.shape[1]):
            i_edge =  tuple_edge.index((explaine_edge[0,i].item(),explaine_edge[1,i].item()))
            edge_weights[i_edge] = 0
            edge_index.remove(i_edge)


        edge_weights = edge_weights.to(device)
        processing_predict = model(feature, edge,  edge_weights = edge_weights)
        if ppi:
            processing_predict = torch.sigmoid(processing_predict)[target_node, :]
        else:
            processing_predict = torch.softmax(processing_predict, dim=1)[target_node, label[target_node]].detach()


    return answer_predict - processing_predict