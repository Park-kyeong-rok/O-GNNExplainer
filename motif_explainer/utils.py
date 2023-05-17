import networkx as nx
from tqdm import tqdm
import copy
import torch
from model import *
import random
from collections import deque

class GenMotif(object):
    def __init__(self, data, type):
        self.type = type
        self.edge_index = []
        self.data = data
        # self.nodes_labels = data.node_labels # For real datasets
        self.vocab = {}
        self.node_id = []
        self.gen_components()



        # self.update_node_id() # For real datasets


    def gen_components(self):
        #g_list는 networkx형태의 그래프들의 리스트
        g_list = self.data.g_list
        h_g = nx.Graph()

        #g는 graph
        for g in tqdm(range(len(g_list)), desc='Gen Components', unit='graph'):
            n_id = []
            edge_id = []
            #모든 사이클을 추출
            mcb = nx.cycle_basis(g_list[g])


            if g==500:
                print(mcb)
            #모든 싸이클을 튜플화
            cycle_edge = []
            mcb_tuple = [tuple([e-1 for e in ele]) for ele in mcb]
            #싸이클 온전히 만들어지는지 학인하기



            for i in mcb_tuple:
                edge_idx = []
                length = len(i)
                for n1_idx in range(length-1):
                    n1 = i[n1_idx]
                    n2 = i[n1_idx+1]
                    edge_idx.append((n1, n2))
                    edge_idx.append((n2, n1))

                #     edge_idx[0].append(n1)
                #     edge_idx[0].append(n2)
                #     edge_idx[1].append(n2)
                #     edge_idx[1].append(n1)
                edge_idx.append((i[0], i[-1]))
                edge_idx.append((i[-1], i[0]))


                if len(edge_idx) != len(set(edge_idx)):
                    print(i)
                    exit()
                cycle_edge.append(edge_idx)


            #mcb_tuple = [tuple(ele) for ele in mcb]
            edges = []
            #circle에 포함되지 않는 edge 추출(edges에)
            # for e in g_list[g].edges():
            #     count = 0
            #     e = tuple([l - 1 for l in e])
            #     for c in mcb_tuple:
            #         if e[0] in set(c) and e[1] in set(c):
            #             count += 1
            #             break
            #
            #     if count == 0:
            #         edges.append(e)

            # edges = list(set(edges))

            ##엣지 넣어주는데
            # for e in edges:
            #     n_id.append(list(e))
            #     edge_id.append([e, (e[1], e[0])])


                ##사이클넣어 주는데
            #print(len(mcb_tuple))
            random_idx = random.sample(range(len(mcb_tuple)), 4000)
            #print(random_idx)

            # if self.type in ['a', 'ab']:
            #     for m in mcb_tuple[random_idx]:
            #         n_id.append(list(m))
            #
            # length = len(mcb_tuple)
            # ###사이클 넣어주는데
            #
            # if self.type in ['a', 'ab']:
            #     for i in cycle_edge[random_idx]:
            #         edge_id.append(i)
            #     #print(len(edge_id))

            if self.type in ['a', 'ab']:
                for i in random_idx:
                    edge_id.append(cycle_edge[i])
                    n_id.append(mcb_tuple[i])
            #합성 넣어주는데
            if self.type in ['b', 'ab']:
                for i in range(length-1):
                    m1 = mcb_tuple[i]


                    for j in range(i+1,length):
                        m2 = mcb_tuple[j]

                        intersect = set(m1)&set(m2)

                        if len(intersect)>=2:
                            intersect_edge = list(set(cycle_edge[i])|set(cycle_edge[j]))

                            intersect_node =  list(set(m1)|set(m2))
                            n_id.append(intersect_node)
                            edge_id.append(intersect_edge)


            self.node_id.append(n_id)
            self.edge_index.append(edge_id)
            print()
            print(f'Using motif number: {len(self.edge_index[0])}')












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
    feature_dim = len(feature_list[0].strip().split(' ')) -2
    n_node = len(feature_list)
    label = torch.zeros((n_node))
    feature = torch.zeros((n_node,feature_dim ))

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



def bfs(graph, start_node, motif_node):
    # 시작 노드를 큐에 삽입
    sub_nods = graph.keys()
    queue = deque([start_node])
    # 시작 노드의 방문 여부를 체크하고 경로를 초기화
    visited = {start_node: True}
    path = {start_node: [start_node]}

    # 큐가 빌 때까지 반복
    while queue:
        # 큐에서 노드를 하나씩 꺼내어 인접한 노드를 모두 방문
        current_node = queue.popleft()
        for adjacent_node in graph[current_node]:
            if adjacent_node not in visited:
                # 인접한 노드를 방문하고 큐에 삽입
                visited[adjacent_node] = True
                path[adjacent_node] = path[current_node] + [adjacent_node]
                queue.append(adjacent_node)
                # 1번 노드를 만났을 때 탐색 종료
                if adjacent_node in motif_node:
                    return path[adjacent_node]
    # 1번 노드를 방문하지 못한 경우, 시작 노드와 1번 노드 사이에 경로가 없음
    return []

def make_graph(data):
    graph = nx.Graph()
    edge_list = open(f'../data/{data}/edge_list.txt', 'r')
    for i in edge_list:
        node1, node2 = map(int, i.strip().split(' '))
        graph.add_edge(node1, node2)
    return graph

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

def fidelity(model, target_node, feature, edge, explaine_edge, label, ppi=None):
    with torch.no_grad():
        answer_predict = model(feature, edge).detach()
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

        for i in explaine_edge:
            i_edge =  tuple_edge.index(i)

            edge_weights[i_edge] = 0
            edge_index.remove(i_edge)


        processing_predict = model(feature, edge, edge_weights = edge_weights)
        if ppi:
            processing_predict = torch.sigmoid(processing_predict)[target_node, :]
        else:
            processing_predict = torch.softmax(processing_predict, dim=1)[target_node, label[target_node]].detach()

    return answer_predict - processing_predict