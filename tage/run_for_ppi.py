import torch
from torch_geometric.utils import k_hop_subgraph
from utils import *
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from copy import copy
from model import *

from tagexplainer import TAGExplainer, MLPExplainer

def get_results(lr, epoch, edge_n, top_k):
    task_n = 0
    thr = 0.5128764
    truth_label = 1
    edge_n = edge_n
    #data_name = 'ppi' + str(task_n)
    data_name = 'ppi_mini'
    #model_name = f'ppi{str(task_n)}_model'
    model_name = f'ppi_mini_model'
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    feature, label, edge_index = load_data(data_name)

    data = Data(x=feature, y=label, edge_index=edge_index)
    data.to(device)
    data_loader = DataLoader(dataset=[data])
#    subgraph = k_hop_subgraph(960, 3, edge_index)[1]



    if agg:
        model_state_dict = torch.load(f'../data/{data_name}/agg_model/{model_name}')
    else:
        model_state_dict = torch.load(f'../data/{data_name}/model/{model_name}')

    model_state_dict_ = copy(model_state_dict)
    for i in model_state_dict_:
        if i.startswith('conv') or i.startswith('bn'):
            del model_state_dict[i]
    if agg:
        model = load_model(data_name, model_name, agg)
        # model_encoder = agg_GCN3_encoder(10,20)
        # model_classifier = agg_GCN3_classifier(8,20)
    if data_name.startswith('ppi'):
        model = load_model(data_name, model_name, agg)
        model_encoder = Net_encoder(50, 100)
        model_classifier = Net_classifier(1, 100)
    else:
        model = load_model(data_name, model_name, agg)
        model_encoder = GCN3_encoder(10, 30)
        model_classifier = GCN3_classifier(8, 30)

    model_classifier.load_state_dict(model_state_dict)
    if agg:
        model_state_dict = torch.load(f'../data/{data_name}/agg_model/{model_name}')
    else:
        model_state_dict = torch.load(f'../data/{data_name}/model/{model_name}')

    for i in model_state_dict_:
        if not (i.startswith('conv')) and not (i.startswith('bn')):
            del model_state_dict[i]

    model_encoder.load_state_dict(model_state_dict)
    model.eval()

    model_encoder.eval()
    model_classifier.eval()

    model.to(device)
    model_encoder.to(device)
    model_classifier.to(device)

    prediction = torch.squeeze((torch.sigmoid(model(data)).detach().to('cpu') >thr).float())

    answer_node_idx = torch.where((label == prediction) == True)[0]
    feature = feature.to(device)
    edge_index = edge_index.to(device)

    enc_explainer = TAGExplainer(model_encoder, embed_dim=300, device=device, explain_graph=False,
                                 grad_scale=0.1, coff_size=0.05, coff_ent=0.002, loss_type='JSE')
    enc_explainer.train_explainer_node(data_loader, batch_size=16, lr=lr, epochs=epoch)

    graph = make_graph(data_name)

    feature = feature.to(device)
    edge_index = edge_index.to(device)


    mlp_explainer = MLPExplainer(model_classifier, device)
    total_fid = 0
    total_spar = 0
    answer_node = 0

    for l, data in enumerate(data_loader):
        for idx in range(0,data.x.shape[0], 10):
            if idx in answer_node_idx and label[idx] !=1-truth_label :
                subgraph = k_hop_subgraph(node_idx=idx, edge_index=edge_index, num_hops=3)[1]
                if subgraph.shape[1] == 0:
                    continue
                answer_node += 1
                print(idx)
                # fid = fidelity(model, 80, feature, edge_index,
                #                torch.LongTensor([[564, 217, 747, 32, 858, 316, 486, 984, 284, 984, 806, 244],
                #                                  [80, 80, 217, 80, 564, 80, 564, 564, 32, 316, 32, 32]]).to(device),
                #                label, ppi=True)
                # print(fid)
                # exit()

                #ground_node = find_baco_gt(graph, idx, label)

                idx = torch.LongTensor([idx]).to(device)

                subgraph = k_hop_subgraph(node_idx=idx, edge_index=edge_index, num_hops=3)


                walks, masks, related_preds = \
                    enc_explainer(data, mlp_explainer, node_idx=idx, top_k=edge_n)

                model.eval()

                important_edge_idx = torch.argsort(masks, descending=True)[:edge_n]
                important_edge = subgraph[1][:, important_edge_idx]

                #
                # for edge_idx in range(important_edge.shape[1]):
                #     node1, node2 = min(important_edge[:, edge_idx].tolist()), max(important_edge[:, edge_idx].tolist())
                #     if label[node1] != 0 and label[node2] != 0 and node1 in ground_node and node2 in ground_node:
                #         acc += 1


                fid = fidelity(model, idx, feature, edge_index, important_edge, label, ppi=True)
                print(fid)
                print(important_edge)
                # if acc == 12:
                #     top1 += 1
                #print(fid)

                subgraph = k_hop_subgraph(idx, 3, edge_index)[1]
                sparsity = 1 - min(important_edge_idx.shape[0],subgraph.shape[1]) / subgraph.shape[1]
                #print(min(important_edge_idx.shape[0],subgraph.shape[1]))
                #print(subgraph.shape[1])
                print('-------')
                total_fid += fid
                total_spar += sparsity


                label.to('cpu')
    print(f'answer_node: {answer_node}')
    return   total_fid.item() / answer_node, total_spar / answer_node

    #         fidelity = related_preds[0]['origin'] - related_preds[0]['maskout']
    #
    #         print(f'explain graph {i} node {node_idx}'+' fidelity %.4f'%fidelity, end='\r')
    #         x_collector.collect_data(masks, related_preds)
    #
    # fid, fid_std = x_collector.fidelity
    # spa, spa_std = x_collector.sparsity

    # print()
    # print(f'Fidelity: {fid:.4f} ±{fid_std:.4f}\n'
    #       f'Sparsity: {spa:.4f} ±{spa_std:.4f}')


#gnn_lr_list = [5e-6, 5e-5, 5e-4, 5e-3,5e-2]
gnn_lr_list = [5e-6]
#epoch = [1, 3, 5, 10]
epoch = [10]
#epoch = [1]
edge_n = [400 ]
#edge_n = [1700]
agg = False
dict_ = dict()
dict__ = dict()

dict___ = dict()
dict____ = dict()
for three in range(1):
    for i in gnn_lr_list:
        for j in epoch:
            for l in edge_n:
                a, b = get_results(i, j, l, 1)

                dict___[f'fid  lr:{i}, epcoh: {j}_, {l}_{three}'] = a
                dict____[f'spar  lr:{i}, epcoh: {j}_, {l}_{three}'] = b
                print('--------------------')
                print(f'fid:{a} spar:{b}')
                print('--------------------')



print(dict_)
print(dict__)
print(dict___)
print(dict____)
