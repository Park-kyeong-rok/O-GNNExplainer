import os
import sys
import torch
import torch_geometric.nn.models
from torch_geometric.utils import k_hop_subgraph
sys.path.append(os.pardir)
from utils import *



def main(gnn_lr, epoch, lap, agg):
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    #model.eval()
    data = 'ppi_mini'
    thr =   0.5128764

    model_name = 'ppi_mini_model'
    method = 'gnnexplainer'

    layer = 3
    gnn_lr = gnn_lr
    model = load_model(data, model_name, agg)
    model.to(device)
    model.eval()

    feature, label, edge = load_data(data)
    feature = feature.to(device)
    label = label.to(device)
    edge = edge.to(device)

    prediction = model(feature, edge)
    #print(model.state_dict())

    predict = torch.squeeze((torch.sigmoid(prediction) > thr).long())

    #print(model.state_dict())
    answer = predict == label

    answer_node_idx = torch.where(answer == True)
    graph = make_graph(data)


    if method == 'gnnexplainer':
        explainer = torch_geometric.nn.models.GNNExplainer(model, epochs=epoch, lr=gnn_lr, num_hops=3, lap=lap, thr=thr)
    total_fid = 0
    answer_node = 0
    total_spar = 0
    for node_idx in range(0, label.shape[0], 10):
        if label[node_idx] != 0 and label[node_idx] != 4 and node_idx in answer_node_idx[0].tolist():
            subgraph = k_hop_subgraph(node_idx, 3, edge)[1]
            if subgraph.shape[1] == 0:
                continue
            answer_node += 1
            important_edge = gnnexplainer_gnn(node_idx, explainer, feature, edge, data, 12)

            important_edge = important_edge.to(device)
            fid = fidelity(model, node_idx, feature, edge, important_edge, label, ppi=True)
            subgraph = k_hop_subgraph(node_idx, 3, edge)[1]
            sparsity = 1 - min(important_edge.shape[1],subgraph.shape[1]) / subgraph.shape[1]
            print(node_idx)
            print(important_edge)
            print(fid)
            #print(important_edge, fid, node_idx)
            total_fid += fid
            total_spar += sparsity
            print(total_spar)
    total_fid /= answer_node
    total_spar /= answer_node

    return total_fid, total_spar
fid = []
spar = []
for i in range(1):
    b, c = main(0.1, 700, 0.1, False)
    fid.append(b.item())
    spar.append(c)
print(fid)
print(spar)