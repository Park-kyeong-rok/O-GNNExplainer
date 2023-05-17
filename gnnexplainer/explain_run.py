import os

import sys

import torch_geometric.nn.models
from torch_geometric.utils import k_hop_subgraph
sys.path.append(os.pardir)
from utils import *



def main(gnn_lr, epoch, lap, top_k ,agg):
    data = 'last_fm'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model_name = '0.001_600_3'
    method = 'gnnexplainer'
    layer = 3
    gnn_lr = gnn_lr


    model = load_model(data, model_name,agg).to(device)

    model.eval()
    feature, label, edge =load_data(data)
    feature = feature.to(device)
    label = label.to(device)
    edge = edge.to(device)

    prediction = model(feature,edge).detach()

    #fidel = fidelity(model, 300, feature, edge, torch.LongTensor([[300, 301], [301, 300]]), label, ppi=None)
    # print(fidel)
    # exit()
    prediction = torch.argmax(prediction, dim=1)

    answer_node_idx = torch.where((label==prediction)==True)




    graph = make_graph(data)

    #ground_node = find_baco_gt(graph, 300, label)
    #plot_nhop(graph, 300, 2, label)


    not_use_orbit = [0, 4]
    if data == 'last_fm':
        not_use_orbit = []
    if method == 'gnnexplainer':
        explainer = torch_geometric.nn.models.GNNExplainer(model,epochs=epoch, lr = gnn_lr, num_hops= 3, lap=lap)
    total_fid = 0
    top1 = 0
    total_recall = 0
    answer_node = 0
    total_acc = 0
    total_spar = 0
    for node_idx in range(0, label.shape[0], 1):
        recall =0
        acc = 0

        if not(label[node_idx] in not_use_orbit) and  node_idx in answer_node_idx[0].tolist() and node_idx%20==0:
            #plot_nhop(graph, node=node_idx, hop=2, label=label)

            #ground_node = find_baco_gt(graph, node_idx, label)
            subgraph = k_hop_subgraph(node_idx, 3, edge)[1]


            answer_node += 1
            answer_edge_list = []
            important_edge = gnnexplainer_gnn(node_idx, explainer, feature, edge, data, top_k)

            spar = 1 - min(subgraph.shape[1], important_edge.shape[1])/subgraph.shape[1]

            # print(important_edge)
            # print(ground_node)
            if data == 'bashapes':
                ground_node = find_baco_gt(graph, node_idx, label)
                ground_edge = 6
            elif data == 'bac':
                ground_node = find_baco_gt(graph, node_idx, label)
                ground_edge = 6
            else:
                ground_node = []
                ground_edge = 6

            #plot_nhop(graph, node_idx, 2, label)
            for edge_idx in range(important_edge.shape[1]):

                node1, node2 = min(important_edge[:,edge_idx].tolist()), max(important_edge[:,edge_idx].tolist())
                if label[node1] !=0 and label[node2] != 0 and node1 in ground_node and node2 in ground_node:
                    acc += 1
                    if (node1, node2) in answer_edge_list:
                        recall += 1

                    answer_edge_list.append((node1, node2))
            #
            # if acc <= 3:
            #     print(label[node_idx])

            if recall == ground_edge:
                top1 += 1
            total_acc += acc/important_edge.shape[1]
            total_recall += recall/(important_edge.shape[1]/2)
            #print(f'{node_idx}: {recall}')
            model.eval()
            fid = fidelity(model, node_idx, feature, edge, important_edge, label)
            print(fid)
            total_fid += fid
            total_spar += spar
            #print(fid, important_edge)


    top1 /= answer_node
    total_fid /= answer_node
    total_spar /= answer_node
    return total_acc/answer_node, top1, total_fid, total_spar

gnn_lr_list = [0.1]

epoch = [600]
laps = [10]

agg = False
dict_ = dict()
dict__ = dict()

dict___ = dict()
dict____ = dict()
for three in range(3):
    for i in gnn_lr_list:
        for j in epoch:
            for l in laps:
                a, b, c, d = main(i, j, l, l, agg)
                dict_[f'top1: {i}_{j}_{l}_{three}'] = b
                dict__[f'acc: {i}_{j}_{l}_{three}'] = a
                dict___[f'fid: {i}_{j}_{l}_{three}'] = c.item()
                dict____[f'spar: {i}_{j}_{l}_{three}'] = d
                print('--------------------')
                print(f'acc{b}, top1{a}, fid{c}, spar{d}')
                print('--------------------')

print(dict_)
print(dict__)
print(dict___)
print(dict____)


