
from utils import *
import torch
import torch_geometric
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import PGExplainer

def main(gnn_lr, epoch, top_k, agg):
    data = 'last_fm'
    device_n = '2'
    device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
    model_name = '0.001_600_3'
    method = 'gnnexplainer'
    layer = 3
    gnn_lr = gnn_lr
    epoch=epoch



    model = load_model(data, model_name,agg).to(device)
    model.eval()

    feature, label, edge =load_data(data)
    feature = feature.to(device)
    label = label.to(device)
    edge = edge.to(device)
    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(n=device_n, epochs=epoch, lr=gnn_lr),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=dict(mode='multiclass_classification',
                          return_type='raw',
                          task_level='node')
    )
    prediction = model(feature, edge).detach()
    prediction = torch.argmax(prediction, dim=1)
    answer_node_idx = torch.where((label == prediction) == True)
    # print(torch.mean((prediction == label).float()))
    # exit()
    graph = make_graph(data)
    model.eval()
    idx = range(label.shape[0])
    label_list = [0,4]
    if data == 'last_fm':
        label_list = []

    for epoch in range(epoch):
        print(epoch)
        for index in idx:  # Indices to train against.
            if not(label[index] in label_list) and index in answer_node_idx[0].tolist() and index%20 ==0:
                print(index)
                #if  index in answer_node_idx[0].tolist():
                loss = explainer.algorithm.train(epoch, model, feature, edge,
                                                target=label, index=index)

    total_fid1 = 0
    total_spar1 = 0
    total_fid2 = 0
    total_spar2 = 0
    total_fid3 = 0
    total_spar3 = 0
    fid_list = []
    spar_list = []
    top1 = 0
    total_acc= 0
    answer_node = 0
    for index in idx:  # Indices to train against.
        if not(label[index] in label_list) and index in answer_node_idx[0].tolist() and index%20 ==0:
            print(index)
            answer_node += 1
            explanation = explainer(feature, edge, target=label, index = index)
            acc = 0


            important_edge = torch.argsort(explanation.edge_mask, descending=True)[:top_k]
            important_edge = explanation.edge_index[:,important_edge]



            if data == 'last_fm':
                ground_node = []
            else:
                ground_node = find_baco_gt(graph, index, label)
            for edge_idx in range(important_edge.shape[1]):
                node1, node2 =  min(important_edge[:,edge_idx].tolist()), max(important_edge[:,edge_idx].tolist())
                if label[node1] !=0 and label[node2] != 0 and node1 in ground_node and node2 in ground_node:
                    acc += 1
            fid1 = fidelity(model, index, feature, edge, important_edge, label)

            if acc == 12:
                top1 += 1
            subgraph = k_hop_subgraph(index, 3, edge)[1]
            sparsity1 = 1 - min(important_edge.shape[1],subgraph.shape[1]) /subgraph.shape[1]
            total_fid1 += fid1
            total_spar1 += sparsity1
            total_acc += acc/12

            # important_edge = torch.argsort(explanation.edge_mask, descending=True)[:1700]
            # important_edge = explanation.edge_index[:,important_edge]
            #
            # fid2 = fidelity(model, index, feature, edge, important_edge, label)
            #
            #
            # subgraph = k_hop_subgraph(index, 3, edge)[1]
            # sparsity2 = 1 - important_edge.shape[1]/subgraph.shape[1]
            # total_fid2 += fid2
            # total_spar2 += sparsity2
            # important_edge = torch.argsort(explanation.edge_mask, descending=True)[:3000]
            # important_edge = explanation.edge_index[:, important_edge]
            #
            # fid3 = fidelity(model, index, feature, edge, important_edge, label)
            #
            # subgraph = k_hop_subgraph(index, 3, edge)[1]
            # sparsity3 = 1 - important_edge.shape[1] / subgraph.shape[1]
            # total_fid3 += fid3
            # total_spar3 += sparsity3
    #fid_list = [total_fid1.detach().to('cpu').item()/answer_node, total_fid2.detach().to('cpu').item()/answer_node, total_fid3.detach().to('cpu').item()/answer_node]
    fid_list = [total_fid1.detach().to('cpu').item() / answer_node]
    #spar_list= [total_spar1/answer_node, total_spar2/answer_node, total_spar3/answer_node]
    spar_list = [total_spar1 / answer_node]
    return total_acc / answer_node, top1/answer_node,fid_list, spar_list

#gnn_lr_list = [0.001, 0.003, 0.01, 0.1]
gnn_lr_list = [0.001]
#epoch = [1, 3, 5, 10, 20]
epoch = [10]
laps = [12, 3000, 5000]
laps =[150]
agg = False
dict_ = dict()
dict__ = dict()

dict___ = dict()
dict____ = dict()
for three in range(5):
    for i in gnn_lr_list:
        for j in epoch:
            for l in laps:
                print(three, i, j, l)
                a, b, c, d = main(i, j, l, agg)
                dict_[f'top1:  lr:{i}, epcoh: {j}, topk: {l}, try:{three}'] = b
                dict__[f'acc:  lr:{i}, epcoh: {j}, topk: {l},try:{three}'] = a
                dict___[f'fid: lr:{i}, epcoh: {j}, topk: {l},try:{three}'] = c
                dict____[f'spar:  lr:{i}, epcoh: {j}, topk: {l},try:{three}'] = d
                print('--------------------')
                print(f'fid:{a},spar:{b}')
                print('--------------------')

print(dict_)
print(dict__)
print(dict___)
print(dict____)


