from arguments import args

import torch
from utils import *
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import numpy as np
from copy import copy
import random
import pickle


data_name = args.data
model_name = args.model_name
thr =args.thr

epochs = 100
save = False
lr = 0.001
agg = False
#a means only cycle, ab mean cycle+ synthetic, b mean only synthetic
type = 'a'
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
feature, label, edge_index = load_data(data_name)

data = Data(x = feature, edge_index = edge_index, label=label)
if agg:
    model_state_dict = torch.load(f'../data/{data_name}/agg_model/{model_name}')
else:
    model_state_dict = torch.load(f'../data/{data_name}/model/{model_name}')
model_state_dict_ = copy(model_state_dict)

for i in model_state_dict_:
    if i.startswith('conv') or i.startswith('bn')  :
        del model_state_dict[i]
if agg:
    model = load_model(data_name, model_name, agg)
    model_encoder = agg_GCN3_encoder(10,30)
    model_classifier = agg_GCN3_classifier(8,30)
else:
    model = load_model(data_name, model_name, agg)
    model_encoder = Net_encoder(50,100)
    model_classifier = Net_classifier(1,100)



model_classifier.load_state_dict(model_state_dict)
if agg:
    model_state_dict = torch.load(f'../data/{data_name}/agg_model/{model_name}')
else:
    model_state_dict = torch.load(f'../data/{data_name}/model/{model_name}')
for i in model_state_dict_:
    if not(i.startswith('conv')) and not(i.startswith('bn')):
        del model_state_dict[i]
model_encoder.load_state_dict(model_state_dict)
model.eval()
model_encoder.eval()
model_classifier.eval()

class GenData(object):
    def __init__(self, g_list, node_labels):
        self.g_list = g_list
        self.node_labels = node_labels

edge_list= []
g_list = []
for i in range(0, edge_index.size()[1]):
    edge_list.append((edge_index[0, i].item()+1, edge_index[1, i].item()+1, {'weight': 0}))

G = nx.Graph()
G.add_edges_from(edge_list)
g_list.append(G)

data_ = GenData(g_list, label)

if not(save):
    graph = GenMotif(data_, type)
    idx_list = open(f'ppi0_motif_graph', 'wb')
    pickle.dump(graph, idx_list)
else:
    print('fuck')
    graph = open(f'ppi0_motif_graph', 'rb')
    graph = pickle.load(graph)

model.eval()
predict = model(data.x, data.edge_index)


predict = torch.squeeze((torch.sigmoid(predict)>thr).long())
#acc = torch.mean((predict == data.label).type(torch.FloatTensor))
tuple_edge = []
for i in range(data.edge_index.shape[1]):
    a, b = data.edge_index[:,i]
    tuple_edge.append((a.item(), b.item()))

total_graph = make_graph(data_name)

#fidel = fidelity(model, 300, feature, data.edge_index,[(300, 301), (301, 300)] , data.label, ppi=None)

# print(len(graph.node_id[0]))
# exit()
total_fid = 0
total_sparse = 0
total_recall = 0
total_top1 = 0
total_anser_node = 0
model_encoder.to(device)
attention_layer = attention(300).train()
total_motif = dict()
total_embedding = dict()
optimizer = torch.optim.Adam(attention_layer.parameters(), lr = lr)
# print(attention_layer.parameters())
# for i in attention_layer.parameters():
#     print(i)
# exit()
for node in range(0, data.x.shape[0], 10):
    print(node)
    # if node not in range(0, 50):
    #     continue
    if predict[node] != data.label[node] or data.label[node] == 0 or data.label[node] == 4:
        continue

    computation_graph = k_hop_subgraph(node, 3,data.edge_index)
    computation_graph_nodes = computation_graph[0].numpy()
    computation_graph_edges = computation_graph[1]

    subgraph = dict()
    for n in computation_graph_nodes:
        subgraph[n.item()] = []
    for e in range(computation_graph_edges.shape[1]):
        n1, n2 = computation_graph_edges[:,e]
        n1, n2 = n1.item(), n2.item()
        subgraph[n1].append(n2)
    length = len(graph.node_id[0])
    motif_embedding = []

    node_motifs = dict()
    for i in range(length):
        motif_nodes = np.array(graph.node_id[0][i])

        # if not (node in motif_nodes):
        #     continue

        motif_edge = copy(graph.edge_index[0][i])
        #print(motif_edge)

        intersect_node = np.intersect1d(motif_nodes, computation_graph_nodes)

        if intersect_node.shape[0] != motif_nodes.shape[0]:
            continue

        if node in motif_nodes:
            path = [node]
        else:
            path = bfs(subgraph,node, motif_nodes)

        feature_zero_node = data.x.clone()

        motif_edge_path = []
        for l in range(1, len(path)):
            n1 = path[l]
            if l !=  len(path)-1:
                feature_zero_node[n1,:] = 0

            motif_edge_path.append((path[l], path[l - 1]))
            motif_edge_path.append((path[l - 1], path[l]))

#
        motif_embedding_feature = model_encoder(feature_zero_node.to(device), torch.LongTensor(motif_edge+motif_edge_path).T.to(device))

        target_node_embedding = model_encoder(data.x.to(device), data.edge_index.to(device))[node, :].detach()

        if  motif_embedding_feature[node, :].detach().tolist() not in motif_embedding:
            node_motifs[i] = []

            motif_embedding.append(motif_embedding_feature[node, :].detach().tolist())
        else:
            idx = list(node_motifs.keys())[motif_embedding.index( motif_embedding_feature[node, :].detach().tolist())]
            node_motifs[idx].append(i)


#    total_motif[node] = node_motifs
    if len(list(node_motifs.keys())) == 0:
        continue


    attention_layer.to(device)

    target_node_embedding = model_encoder(data.x.to(device), data.edge_index.to(device))[node,:].detach()
    motif_embedding = torch.Tensor(motif_embedding).to(device)

    model_classifier.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()


    for epoch in range(epochs):
        new_embedding_representation = attention_layer(target_node_embedding, motif_embedding)
        #print(attention_layer.score)

        #print(new_embedding_representation)
        #print('------')
        y =model_classifier(new_embedding_representation)

        #print(new_embedding_representation)
        #print(data.label[node])
        loss = criterion(y, torch.unsqueeze(data.label[node].to(device).float(), dim=-1))


        optimizer.zero_grad()
        loss.backward()
        #print(attention_layer.weight.grad)
        optimizer.step()
        #print(loss)
    print('-----')

for node in range(0, data.x.shape[0], 10):
    # if node not in range(0, 50):
    #     continue
    if predict[node] != data.label[node] or data.label[node] == 0 or data.label[node] == 4:
        continue
    computation_graph = k_hop_subgraph(node, 3, data.edge_index)
    if computation_graph[1].shape[1] ==0 :
        continue
    target_node_embedding = model_encoder(data.x.to(device), data.edge_index.to(device))[node, :].detach()
    motif_embedding = []
    node_motifs = dict()
    computation_graph = k_hop_subgraph(node, 3,data.edge_index)
    computation_graph_nodes = computation_graph[0].numpy()
    computation_graph_edges = computation_graph[1]

    subgraph = dict()
    for n in computation_graph_nodes:
        subgraph[n.item()] = []
    for e in range(computation_graph_edges.shape[1]):
        n1, n2 = computation_graph_edges[:,e]
        n1, n2 = n1.item(), n2.item()
        subgraph[n1].append(n2)

    for i in range(length):
        motif_nodes = np.array(graph.node_id[0][i])

        # if not (node in motif_nodes):
        #     continue

        motif_edge = copy(graph.edge_index[0][i])
        # print(motif_edge)

        intersect_node = np.intersect1d(motif_nodes, computation_graph_nodes)

        if intersect_node.shape[0] != motif_nodes.shape[0]:
            continue

        if node in motif_nodes:
            path = [node]
        else:
            path = bfs(subgraph, node, motif_nodes)

        feature_zero_node = data.x.clone()

        motif_edge_path = []
        for l in range(1, len(path)):
            n1 = path[l]
            if l != len(path) - 1:
                feature_zero_node[n1, :] = 0

            motif_edge_path.append((path[l], path[l - 1]))
            motif_edge_path.append((path[l - 1], path[l]))

        #
        motif_embedding_feature = model_encoder(feature_zero_node.to(device),
                                                torch.LongTensor(motif_edge + motif_edge_path).T.to(device))

        target_node_embedding = model_encoder(data.x.to(device), data.edge_index.to(device))[node, :].detach()

        if motif_embedding_feature[node, :].detach().tolist() not in motif_embedding:
            node_motifs[i] = []

            motif_embedding.append(motif_embedding_feature[node, :].detach().tolist())
        else:
            idx = list(node_motifs.keys())[motif_embedding.index(motif_embedding_feature[node, :].detach().tolist())]
            node_motifs[idx].append(i)
    motif_embedding = torch.Tensor(motif_embedding).to(device)
    if len(list(node_motifs.keys())) == 0:
        continue
    new_embedding_representation = attention_layer(target_node_embedding, motif_embedding)
    max_idx = torch.argmax(attention_layer.score, dim=0)
    #print(attention_layer.score)

    max_motif = [list(node_motifs.keys())[max_idx]]+node_motifs[list(node_motifs.keys())[max_idx]]
    # print(node_motifs)
    # print(max_motif)

    fidelities = []
    computation_graph = k_hop_subgraph(node, 3,data.edge_index)
    computation_graph_nodes = computation_graph[0].numpy()
    computation_graph_edges = computation_graph[1]

    subgraph = dict()
    for n in computation_graph_nodes:
        subgraph[n.item()] = []
    for e in range(computation_graph_edges.shape[1]):
        n1, n2 = computation_graph_edges[:,e]
        n1, n2 = n1.item(), n2.item()
        subgraph[n1].append(n2)

    for m in max_motif:
        # print('aaa')
        motif_nodes = np.array(graph.node_id[0][m])
        # print(motif_nodes)
        # print(graph.edge_index[0][m])

        if node in motif_nodes:
            path = [node]
        else:
            path = bfs(subgraph,node, motif_nodes)

        feature_zero_node = data.x.clone()

        motif_edge_path = []
        for l in range(1, len(path)):
            n1 = path[l]
            if l !=  len(path)-1:
                feature_zero_node[n1,:] = 0

            motif_edge_path.append((path[l], path[l - 1]))
            motif_edge_path.append((path[l - 1], path[l]))
        # print(motif_edge_path + graph.edge_index[0][m])
        # print('fuck')
        # print(graph.edge_index[0][m])
        # exit()
        #fidel = fidelity(model, node, feature, data.edge_index,graph.edge_index[0][m] , data.label, ppi=None)
        fidel = fidelity(model, node, feature_zero_node, data.edge_index, motif_edge_path+graph.edge_index[0][m], data.label, ppi=True)
        fidelities.append((fidel,graph.edge_index[0][m]))
    #print(fidelities)
    random.shuffle(fidelities)

    max_fid = max(fidelities)[0]

    predicted_edge = max(fidelities)[1]
    print(max_fid)
    print(predicted_edge)
    # ground_node = find_baco_gt(total_graph, node, data.label)
    # ground_edge = []
    # for node_idx1 in range(len(ground_node)-1):
    #     for node_idx2 in range(node_idx1, len(ground_node)):
    #         if (ground_node[node_idx1], ground_node[node_idx2]) in tuple_edge:
    #             ground_edge.append((ground_node[node_idx1], ground_node[node_idx2]))
    #             ground_edge.append((ground_node[node_idx2], ground_node[node_idx1]))
    # #print(ground_edge)
    total_anser_node += 1
    total_fid += max_fid
    # recall = len(set(ground_edge) & set(predicted_edge)) / len(ground_edge)
    # if recall == 1:
    #     total_top1 += 1
    # total_recall += recall

    total_sparse += 1 - min(len(predicted_edge),computation_graph_edges.shape[1]) / computation_graph_edges.shape[1]

    # ground_node = find_baco_gt(total_graph, node, data.label)
    # print(ground_node)
    print(node)

print(f'fid: {total_fid / total_anser_node}, sparse: {total_sparse / total_anser_node}')
txt = open('result.txt', 'a')
txt.write(f'{data_name}, fid:{total_fid.item() / total_anser_node}, spar: {total_sparse / total_anser_node}\n')

