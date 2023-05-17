import torch
from model import *
from collections import deque
import networkx as nx
import torch_geometric

def fidelity(model, target_node, feature, edge, explaine_edge, label, ppi=None):
    with torch.no_grad():
        device = feature.device

        data = torch_geometric.data.Data(x = feature, edge_index=edge)
        data.to(device)
        label = label.to(device)

        answer_predict = model(data).detach()

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
        processing_predict = model(data,  edge_weights = edge_weights)
        if ppi:
            processing_predict = torch.sigmoid(processing_predict)[target_node, :]
        else:
            processing_predict = torch.softmax(processing_predict, dim=1)[target_node, label[target_node]].detach()


    return answer_predict - processing_predict

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
    if data.startswith('ppi'):
        feature = torch.zeros((n_node, 50))
    else:
        feature = torch.zeros((n_node, 7842))


    for features in feature_list:
        features = list(map(float, features.strip().split()))
        node, features_, label_ = features[0], features[1:-1], features[-1]

        label[int(node)] = label_
        feature[int(node), :] = torch.FloatTensor(features_)


    return feature, label.long(), edge.long()


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

def make_graph(data):
    graph = nx.Graph()
    edge_list = open(f'../data/{data}/edge_list.txt', 'r')
    for i in edge_list:
        node1, node2 = map(int, i.strip().split(' '))
        graph.add_edge(node1, node2)
    return graph