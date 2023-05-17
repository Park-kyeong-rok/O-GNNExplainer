from utils import *
from arguments import args

import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

x, label, edge = load_data(args.data)


print(x)

data = Data(x, edge)

u = 28
v = 384
has_edge = (data.edge_index[0] == u) & (data.edge_index[1] == v)

if torch.sum(has_edge) > 0:
    print("엣지가 존재합니다.")
else:
    print("엣지가 존재하지 않습니다.")

edge = torch.LongTensor([(384, 839), (839, 384), (661, 839), (839, 661), (29, 384), (384, 29), (661, 29), (29, 661), (661, 30), (29, 30), (384, 30), (839, 30),
 (30, 661), (30, 29), (30, 384), (30, 839)] )
edge = edge.T
edge2 = torch.LongTensor([[514, 430, 127, 268, 268, 238, 850, 135, 840, 346, 422, 512],
        [430, 514, 430, 430, 514, 430, 430, 430, 430, 430, 430, 127]])

edge2 = torch.LongTensor([(854, 29), (29, 854), (29, 28), (28, 29), (854, 28), (28, 854)]).T

#motif = [(436, 827), (827, 436), (827, 135), (135, 827), (135, 430), (430, 135), (430, 268), (268, 430), (436, 268), (268, 436)]
x = torch.unique(edge)
edge = torch.concatenate([edge, edge2], dim= 1)
x = torch.unique(edge)
print(x)




data = Data(x, edge)
def to_networkx(data):
    edge_index = data.edge_index
    x = data.x

    edge_list = edge_index.t().tolist()
    node_labels = {i: x[i].item() for i in range(x.size(0))}

    G = nx.Graph()
    G.add_edges_from(edge_list)
    nx.set_node_attributes(G, node_labels, 'label')

    return G

def visualize_graph(G):
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')

    plt.figure(figsize=(8, 6))
    nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', node_size=2)
    nx.draw_networkx_labels(G, pos, labels=labels)

    plt.axis('off')
    plt.show()

# NetworkX 그래프로 변환
graph = to_networkx(data)

# 그래프 시각화
visualize_graph(graph)