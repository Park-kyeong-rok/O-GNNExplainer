
import torch
from utils import *
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import numpy as np
from copy import copy
import pickle

data_name = 'ppi0'
model_name = 'ppi0_model'
thr =0.643543541431427
epochs = 100
lr = 0.01
agg = False
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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
    model_encoder = agg_GCN3_encoder(10,20)
    model_classifier = agg_GCN3_classifier(8,20)
else:
    model = load_model(data_name, model_name, agg)
    model_encoder = Net_encoder(50,200)
    model_classifier = Net_classifier(1,200)


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
for i in range(edge_index.size()[1]):
    edge_list.append((edge_index[0, i].item()+1, edge_index[1, i].item()+1, {'weight': 0}))

G = nx.Graph()
G.add_edges_from(edge_list)
g_list.append(G)
data_ = GenData(g_list, label)
graph = GenMotif(data_)
txt = open(f'{data_name}.pickle', 'wb')
pickle.dump(graph, txt)
a = open(f'{data_name}.pickle', 'rb')
graph = pickle.load(a)


predict = model(data.x, data.edge_index)
model.eval()

predict = torch.squeeze((torch.sigmoid(predict)>thr).long())
print(torch.sum(predict))
#acc = torch.mean((predict == data.label).type(torch.FloatTensor))
tuple_edge = []
for i in range(data.edge_index.shape[1]):
    a, b = data.edge_index[:,i]
    tuple_edge.append((a.item(), b.item()))

total_graph = make_graph(data_name)


# print(len(graph.node_id[0]))
# exit()
total_fid = 0
total_sparse = 0
total_recall = 0
total_top1 = 0
total_anser_node = 0
model_encoder.to(device)
for node in range(0, data.x.shape[0], 5):
    attention_layer = attention(600)
    if predict[node] != data.label[node] or data.label[node] == 0 or data.label[node] == 4:
        continue
    computation_graph = k_hop_subgraph(node, 3,data.edge_index)
    computation_graph_nodes = computation_graph[0].numpy()
    computation_graph_edges = computation_graph[1]
    print('yes')
    subgraph = dict()
    for n in computation_graph_nodes:
        subgraph[n.item()] = []
    for e in range(computation_graph_edges.shape[1]):
        n1, n2 = computation_graph_edges[:,e]
        n1, n2 = n1.item(), n2.item()
        subgraph[n1].append(n2)
    print('yes2')
    length = len(graph.node_id[0])
    motif_embedding = []
    print(length)
    node_motifs = dict()
    for i in range(length):

        motif_nodes = np.array(graph.node_id[0][i])

        motif_edge = copy(graph.edge_index[0][i])


        intersect_node = np.intersect1d(motif_nodes, computation_graph_nodes)

        if intersect_node.shape[0] != motif_nodes.shape[0]:
            continue

        if node in motif_nodes:
            path = [node]
        else:
            path = bfs(subgraph,node, motif_nodes)

        feature_zero_node = data.x.clone()
        #print(path)
        #print(motif_edge)

        for l in range(1, len(path)):
            n1 = path[l]
            if l !=  len(path)-1:
#                print('y')
                feature_zero_node[n1,:] = 0
            motif_edge.append((path[l],path[l-1] ))
            motif_edge.append((path[l-1], path[l]))
        #print('yes3')
#        print(motif_edge)

        # edge_weight = torch.zeros((data.edge_index.shape[1]))
        # for e in motif_edge:
        #     e_idx = tuple_edge.index(e)
        #     edge_weight[e_idx] = 1

        motif_embedding_feature = model_encoder(feature_zero_node.to(device), torch.LongTensor(motif_edge).T.to(device))
        #motif_embedding.append(motif_embedding_feature)

        #print(motif_embedding_feature[node, :])
        #print(target_node_embedding[node, :])
        #print(motif_embedding_feature[node, :].detach().tolist())

        if  motif_embedding_feature[node, :].detach().tolist() not in motif_embedding:
            node_motifs[i] = []

            motif_embedding.append(motif_embedding_feature[node, :].detach().tolist())
        else:
            idx = list(node_motifs.keys())[motif_embedding.index( motif_embedding_feature[node, :].detach().tolist())]
            node_motifs[idx].append(i)

        #print(model(feature_zero_node, data.edge_index, edge_weights = edge_weight)[node, :])
        #print(model(data.x, data.edge_index)[node,:])
        # if len(motif_edge) != len(set(motif_edge)):
        #     print(motif_edge)
        #     print(path)
        #     print(motif_nodes)
        #     print(graph.edge_index[0][i])
        #     for l in range(1, len(path)):
        #         n1 = path[l]
        #         if l != len(path) - 1:
        #             feature_zero_node[n1, :] = 0
        #         motif_edge.append((path[l], path[l - 1]))
        #         motif_edge.append((path[l - 1], path[l]))
        #         print((path[l], path[l - 1]))
        #     exit()


    attention_layer.to(device)

    target_node_embedding = model_encoder(data.x.to(device), data.edge_index.to(device))[node,:].detach()
    motif_embedding = torch.Tensor(motif_embedding).to(device)
    model_classifier.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(attention_layer.parameters(), lr = lr)

    for epoch in range(epochs):
        new_embedding_representation = attention_layer(target_node_embedding, motif_embedding)
        y =model_classifier(new_embedding_representation)
        loss = criterion(y, torch.unsqueeze(data.label[node].float().to(device),dim=-1))
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('-----')
    max_idx = torch.argmax(attention_layer.score, dim=0)
    max_motif = [list(node_motifs.keys())[max_idx]]+node_motifs[list(node_motifs.keys())[max_idx]]
    fidelities = []

    for m in max_motif:

        fidel = fidelity(model, node, feature, data.edge_index,graph.edge_index[0][m] , data.label, ppi=True)
        fidelities.append((fidel,graph.edge_index[0][m]))

    max_fid = max(fidelities)[0]
    predicted_edge = max(fidelities)[1]




    total_anser_node += 1
    total_fid += max_fid




    total_sparse += 1 - len(predicted_edge)/computation_graph_edges.shape[1]

    #ground_node = find_baco_gt(total_graph, node, data.label)
    #print(ground_node)
    print(node)

print(f'recall: {total_recall/total_anser_node}, fid: {total_fid/total_anser_node}, sparse: {total_sparse/total_anser_node}, top1: {total_top1/total_anser_node}')


    # print(motif_embedding)
    #
    #
    # print(motif_embedding.shape)




