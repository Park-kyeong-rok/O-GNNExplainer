from utils import *
import torch
import torch_geometric
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import PGExplainer

def main(gnn_lr, epoch, edge_n, agg):
    task_n = 0
    thr =0.5128764
    target_label = 1
    #data = 'ppi' + str(task_n)
    data = 'ppi_mini'
    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
    device_n = '2'
    #model_name =  f'ppi{str(task_n)}_model'
    model_name = 'ppi_mini_model'
    method = 'gnnexplainer'
    layer = 3
    gnn_lr = gnn_lr
    epoch=epoch
    edge_n = edge_n



    model = load_model(data, model_name,agg).to(device)

    model.eval()
    feature, label, edge =load_data(data)
    feature = feature.to(device)
    label = label.to(device)
    edge = edge.to(device)
    # explainer = Explainer(
    #     model=model,
    #     algorithm=PGExplainer(epochs=epoch, lr=gnn_lr),
    #     explanation_type='phenomenon',
    #     edge_mask_type='object',
    #     binary_thr = thr,
    #     model_config=dict(mode='binary_classification',
    #                       return_type='raw',
    #                       task_level='node')
    # )
    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(n=device_n, epochs=epoch, lr=gnn_lr),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=dict(mode='binary_classification',
                          return_type='raw',
                          task_level='node')
    )
    prediction = torch.squeeze(torch.sigmoid(model(feature, edge).detach().to('cpu')) > thr).float().to(device)
    answer_node_idx = torch.where((label == prediction) == True)[0]
    print()

    graph = make_graph(data)
    model.eval()
    idx = range(label.shape[0])

    for epoch in range(epoch):
        for index in range(0,label.shape[0], 5):
            #index = index.item()# Indices to train against.
            #if label[index] != 1-target_label and label[index] != 4 and index in answer_node_idx.tolist():
            if index in answer_node_idx.tolist():
                print(label)
                loss = explainer.algorithm.train(epoch, model, feature, edge,
                                                target=label, index=index)
                if index == 10:
                    print(f'{index}, {loss}')
    total_fid = 0
    total_spar = 0
    model.eval()
    answer_node = 0
    for index in range(0,label.shape[0], 10) :  # Indices to train against.
        if label[index] !=  1-target_label and label[index] != 4 and index in answer_node_idx.tolist():
            print(index)
            subgraph = k_hop_subgraph(index, 3, edge)[1]
            if subgraph.shape[1] == 0:
                continue
        #if index in answer_node_idx.tolist():
            answer_node += 1
            print(answer_node)
            explanation = explainer(feature, edge, target=label, index = index)



            important_edge = torch.argsort(explanation.edge_mask, descending=True)[:edge_n]
            #print('aaa', explanation.edge_mask[important_edge[0]], explanation.edge_mask[important_edge[5]])
            important_edge = explanation.edge_index[:,important_edge]



            #ground_node = find_baco_gt(graph, index, label)
            # for edge_idx in range(important_edge.shape[1]):
            #     node1, node2 =  min(important_edge[:,edge_idx].tolist()), max(important_edge[:,edge_idx].tolist())
            #     if label[node1] !=0 and label[node2] != 0 and node1 in ground_node and node2 in ground_node:
            #         acc += 1
            #print(important_edge)
            fid = fidelity(model, index, feature, edge, important_edge, label, True)
            #print(fid)
            # if acc == 12:
            #     top1 += 1
            subgraph = k_hop_subgraph(index, 3, edge)[1]
            sparsity = 1 - min(edge_n,subgraph.shape[1]) /subgraph.shape[1]
            print(fid)
            print(important_edge)
            print('-------')
            total_fid += fid
            total_spar += sparsity
    print(f'answer_node: {answer_node}')

    return  total_fid.detach().to('cpu').item()/answer_node, total_spar/answer_node

#gnn_lr_list = [0.001, 0.003, 0.01, 0.1]
gnn_lr_list = [0.001]
#epoch = [1, 3, 5, 10, 20]
epoch = [10]
laps = [70 ]
#laps = [3000]
agg = False
dict_ = dict()
dict__ = dict()

dict___ = dict()
dict____ = dict()
for three in range(1):
    for i in gnn_lr_list:
        for j in epoch:
            for l in laps:
                a, b  = main(i, j, l, agg)

                dict___[f'fid: {i}_{j}_{l}_{three}'] = a
                dict____[f'spar: {i}_{j}_{l}_{three}'] = b
                print('--------------------')
                print(f'spar:{b}, fid:{a}')
                print('--------------------')


print(dict_)
print(dict__)
print(dict___)
print(dict____)


#main(1, 2, 3, False)