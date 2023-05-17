import torch
from utils import *
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import numpy as np
from copy import copy
import pickle

data_name = 'last_fm'
model_name = '0.001_600_3'
epochs = 50
lr = 0.001
agg = False
save = False
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
    model_encoder = GCN3_encoder(7842,30)
    model_classifier = GCN3_classifier(18,30)


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
for i in range(0, edge_index.size()[1], 5):
    edge_list.append((edge_index[0, i].item()+1, edge_index[1, i].item()+1, {'weight': 0}))

G = nx.Graph()
G.add_edges_from(edge_list)
g_list.append(G)

data_ = GenData(g_list, label)
if not(save):
    graph = GenMotif(data_, type)
    idx_list = open(f'last_fm_motif_graph', 'wb')
    pickle.dump(graph, idx_list)
else:
    print('fuck')
    graph = open(f'last_fm_motif_graph', 'rb')
    graph = pickle.load(graph)


model.eval()
predict = model(data.x, data.edge_index)


predict = torch.argmax(torch.softmax(predict, dim = 1),dim=1)
#acc = torch.mean((predict == data.label).type(torch.FloatTensor))
tuple_edge = []
model.to(device)
model_encoder.to(device)
model_classifier.to(device)
for i in range(data.edge_index.shape[1]):
    a, b = data.edge_index[:,i]
    tuple_edge.append((a.item(), b.item()))

total_graph = make_graph(data_name)
label_list = [0, 4]
if data_name == 'last_fm':
    label_list = []

motif_idx_list = dict({})
motif_embedding_list = dict({})

# print(len(graph.node_id[0]))
# exit()
data_path = f'{data_name}'
attention_layer = attention(10)
if save:
    idx_list = open(f'{data_path}/idx_list', 'rb')
    embedding_list = open(f'{data_path}/embedding_list', 'rb')
    motif_idx_list = pickle.load(idx_list)
    motif_embedding_list = pickle.load(embedding_list)


else:
    for node in range(0, data.x.shape[0]):
        if predict[node] != data.label[node] or not(data.label[node] in label_list):
            continue
        print(node)
        computation_graph = k_hop_subgraph(node, 3,data.edge_index)
        computation_graph_nodes = computation_graph[0].numpy()
        computation_graph_edges = computation_graph[1]

        subgraph = dict()
        for n in computation_graph_nodes:
            subgraph[n.item()] = []
        for e in range(computation_graph_edges.shape[1]):
            n1, n2 = computation_graph_edges[:,e]
            n1, n2 = n1.item(), n2.item()

        #각 노드의 인덱스에서 0은 motfi idx, 1은 path
        motif_idx_list[node] =dict()
        motif_embedding_list[node] = []
        length = len(graph.node_id[0])
        motif_embedding = []
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
            motif_edge_path = []
            for l in range(1, len(path)):
                n1 = path[l]
                if l !=  len(path)-1:
    #                print('y')
                    feature_zero_node[n1,:] = 0
                #motif_edge.append((path[l],path[l-1] ))
                #motif_edge.append((path[l-1], path[l]))
                motif_edge_path.append((path[l], path[l - 1]))
                motif_edge_path.append((path[l - 1], path[l]))


            motif_embedding_feature = model_encoder(feature_zero_node.to(device), torch.LongTensor(motif_edge+motif_edge_path).T.to(device))


            if  motif_embedding_feature[node, :].detach().tolist() not in motif_embedding_list[node]:
                motif_idx_list[node][i] = []

                motif_embedding_list[node].append(motif_embedding_feature[node, :].detach().tolist())
            else:
                # print(list(motif_idx_list[node].keys()))
                # print(motif_embedding_list[node].index( motif_embedding_feature[node, :].detach().tolist()))
                idx = list(motif_idx_list[node].keys())[motif_embedding_list[node].index( motif_embedding_feature[node, :].detach().tolist())]
                motif_idx_list[node][idx].append(i)
    idx_list = open(f'{data_path}/idx_list', 'wb')
    embedding_list = open(f'{data_path}/embedding_list', 'wb')
    pickle.dump(motif_idx_list, idx_list)
    pickle.dump(motif_embedding_list, embedding_list)


for epoch in range(epochs):
    for node in range(0, data.x.shape[0], 5):
        if predict[node] != data.label[node] or data.label[node] == 0 or data.label[node] == 4:
            continue
        attention_layer.to(device)
        target_node_embedding = model_encoder(data.x.to(device), data.edge_index.to(device))[node,:].detach()
        print(motif_embedding[node])
        print(motif_idx_list[node])
        exit()