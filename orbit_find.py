import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
from itertools import combinations, permutations
from utils import *
from arguments import args
import random

def fidelity(model, target_node, feature, edge, explaine_edge, label):
    with torch.no_grad():
        answer_predict = model(feature, edge).detach()
        if args.model_name.startswith('ppi'):
            answer_predict = torch.sigmoid(answer_predict)[target_node, :].item()

        else:
            answer_predict = torch.softmax(answer_predict, dim=1)[target_node, label[target_node]].item()
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
        if args.model_name.startswith('ppi'):
            processing_predict = torch.sigmoid(processing_predict)[target_node, :].item()
        else:
            processing_predict = torch.softmax(processing_predict, dim=1)[target_node, label[target_node]].item()

    return answer_predict - processing_predict

# def fidelity(model, target_node, feature, edge, explaine_edge, label):
#     with torch.no_grad():
#         answer_predict = model(feature, edge).detach()
#         if args.model_name.startswith('ppi'):
#             answer_predict = torch.sigmoid(answer_predict)[target_node, :].item()
#
#         else:
#             answer_predict = torch.softmax(answer_predict, dim=1)[target_node, label[target_node]].item()
#         tuple_edge= []
#         edge_index = [i for i in range(edge.shape[1])]
#         edge_weights = torch.ones(edge.size(1))
#
#
#         node = []
#         for i, j in explaine_edge:
#             node.append(i)
#             node.append(j)
#         node = list(set(node))
#
#         node_idx = node.index(target_node)
#         explaine_edge = torch.LongTensor(explaine_edge).T
#
#
#         # exit()
#         # for i in range(edge.shape[1]):
#         #     a, b = edge[:,i]
#         #     tuple_edge.append((a.item(), b.item()))
#
#         # for i in explaine_edge:
#         #     i_edge =  tuple_edge.index(i)
#         #     edge_weights[i_edge] = 0
#         #     edge_index.remove(i_edge)
#
#         processing_predict = model(feature, explaine_edge)
#         if args.model_name.startswith('ppi'):
#             processing_predict = torch.sigmoid(processing_predict)[target_node, :].item()
#         else:
#             processing_predict = torch.softmax(processing_predict, dim=1)[target_node, label[target_node]].item()
#         print(answer_predict - processing_predict)
#         exit()
#     return answer_predict - processing_predict

# def fidelity(model, target_node, feature, edge, explaine_edge, label):
#     answer_predict = model(feature, edge)
#     if args.model_name.startswith('ppi'):
#         print(torch.sigmoid(answer_predict).shape)
#         answer_predict = torch.sigmoid(answer_predict)[target_node,:]
#         print(answer_predict)
#
#     else:
#         answer_predict = torch.softmax(answer_predict, dim=1)[target_node, label[target_node]]
#     tuple_edge = []
#     edge_index = [i for i in range(edge.shape[1])]
#     for i in range(edge.shape[1]):
#         a, b = edge[:, i]
#         tuple_edge.append((a.item(), b.item()))
#
#     for i in explaine_edge:
#         i_edge = tuple_edge.index(i)
#         edge_index.remove(i_edge)
#
#     new_edge = edge[:, edge_index]
#     processing_predict = model(feature, new_edge)
#     processing_predict = torch.softmax(processing_predict, dim=1)[target_node, label[target_node]]
#
#     return answer_predict - processing_predict

def find_2_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n=1000 ):
    neigbor = list(graph.neighbors(node.item()))
    graph_edge = graph.edges()
    combination = list(combinations(neigbor, 2))
    n = 0
    random.shuffle(combination)
    max_fider = -500
    predict_edge = []
    for j in combination:
        a, b = j
        if (a, b) not in graph_edge:
            ground_edge = [(a, node), (node, a), (node, b), (b, node)]
            fider = fidelity(model,node, feature, edge,ground_edge, label)
            n += 1
            if max_fider < fider:
                predict_edge = ground_edge
                max_fider = fider
            if n == choice_n:
                return max_fider, predict_edge
    if predict_edge == []:
        print('i dont search 2 orbit')
        return 0, []
    return max_fider, predict_edge

def find_3_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n=1000 ):
    neigbor = list(graph.neighbors(node.item()))
    graph_edge = graph.edges()
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    max_fider = -500
    predict_edge = []
    n = 0
    for j in combination:
        a, b = j
        if (a, b) in graph_edge:

            ground_edge = [(a, node), (node, a), (node, b), (b, node), (a, b), (b, a)]
            fider = fidelity(model,node, feature, edge,ground_edge, label)
            n += 1
            if max_fider < fider:
                predict_edge = ground_edge
                max_fider = fider
            if choice_n == n:
                return max_fider, predict_edge
    if predict_edge == []:
        print('i dont search 3 orbit')
        return 0, []
    return max_fider, predict_edge

def find_16_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    max_fider = -500
    predict_edge = []
    combination = list(combinations(neigbor, 4))
    n = 0
    random.shuffle(combination)
    for a, b, c,d in combination:
        if edge_matrix[a, b] == 1 or edge_matrix[b, c] == 1 or edge_matrix[c, d] == 1 or edge_matrix[d, a] == 1 or edge_matrix[a, c] == 1 or edge_matrix[b, d] == 1:
            continue
        n += 1
        ground_edge = [(node, a), (a, node), (node, b), (b, node), (node, c), (c, node), (d, node), (node, d)]
        fider = fidelity(model, node, feature, edge, ground_edge, label)
        if max_fider < fider:
            predict_edge = ground_edge
            max_fider = fider
        if choice_n == n:
            return max_fider, predict_edge
    if predict_edge == []:
        return 0, []

    return max_fider, predict_edge
def find_4_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    graph_edge = graph.edges()
    max_fider = -500
    predict_edge = []
    n = 0
    random.shuffle(neigbor)
    for j in neigbor:

        combination = list(combinations(list(graph.neighbors(j)),2))
        random.shuffle(combination)
        for (a, b) in combination:
            if a != node.item() and b != node.item() and edge_matrix[a,b]==0 and edge_matrix[a,node]==0 and edge_matrix[b, node]==0:

                n+=1
                ground_edge = [(j, node), (node, j), (j, b), (b, j), (a, j), (j, a)]
                fider = fidelity(model,node, feature, edge,ground_edge, label)
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if choice_n == n:
                    return max_fider, predict_edge
    if predict_edge == []:
        print('i dont search 4 orbit')
        return 0, []

    return max_fider, predict_edge
def find_5_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    max_fider = -500
    predict_edge = []
    combination = list(combinations(neigbor, 3))
    n = 0
    random.shuffle(combination)
    for a, b, c in combination:
        if edge_matrix[a, b] == 1 or edge_matrix[b, c] ==1 or edge_matrix[c, a] == 1:
            continue
        n += 1
        ground_edge = [(a, node), (node, a), (node, b), (b, node), (c, node), (node, c)]
        fider = fidelity(model,node, feature, edge,ground_edge, label)
        if max_fider < fider:
            predict_edge = ground_edge
            max_fider = fider
        if choice_n == n:
            return max_fider, predict_edge
    if predict_edge == []:
        return 0, []

    return max_fider, predict_edge
def find_6_orbit(graph, node, label, model, feature, edge, edge_matrix,choice_n = 1000 ):
    neigbor = list(graph.neighbors(node.item()))
    max_fider = -500
    predict_edge = []
    n = 0
    random.shuffle(neigbor)
    for a in neigbor:
        a_neighbor = list(graph.neighbors(a))
        a_neighbor.remove(node)
        random.shuffle(a_neighbor)
        for b in a_neighbor:
            if edge_matrix[b, node] == 1:
                continue
            b_neighbor = list(graph.neighbors(b))
            b_neighbor.remove(a)
            random.shuffle(b_neighbor)
            if node in b_neighbor:
                b_neighbor.remove(node)
            for c in b_neighbor:
                if edge_matrix[c, a] == 1 or edge_matrix[c, node] == 1:
                    continue
                ground_edge = [(node, a), (a, node), (a, b), (b, a), (b, c), (c, b)]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                n += 1
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if choice_n == n :
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []

    return max_fider, predict_edge
def find_7_orbit(graph, node, label, model, feature, edge ,edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    graph_edge = graph.edges()
    combination = list(combinations(neigbor, 2))
    max_fider = -500
    predict_edge = []
    combination2 = []
    n = 0
    random.shuffle(combination)

    for j in combination:
        a, b = j
        if not(a, b) in graph_edge:
            combination2.append((a, b))
    random.shuffle(combination2)
    for node1, node2 in combination2:

        node2_neighbor = list(graph.neighbors(node2))
        random.shuffle(node2_neighbor)
        for j in node2_neighbor:
            if edge_matrix[j, node] == 0 and edge_matrix[j, node1] == 0:
                ground_edge = [(node, node1), (node1, node), (node, node2), (node2, node), (j, node2), (node2, j) ]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                n += 1
                if max_fider < fider:
                    max_fider = fider
                    predict_edge = ground_edge
                if n == choice_n:
                    return max_fider, predict_edge
    for node1, node2 in combination2:

        node1_neighbor = list(graph.neighbors(node1))
        random.shuffle(node1_neighbor)
        for j in node1_neighbor:
            if edge_matrix[j, node] == 0 and edge_matrix[j, node2] == 0:
                n += 1
                ground_edge = [(node, node1), (node1, node), (node, node2), (node2, node), (j, node1), (node1, j) ]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    max_fider =fider
                    predict_edge = ground_edge
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        print('i dont search 7 orbit')
        return 0, []
    return max_fider, predict_edge

def find_54_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    max_fider = -50000
    predict_edge = []
    node = node.item()
    comb = list(combinations(neigbor, 4))
    random.shuffle(comb)
    n = 0
    for ida, idb, idc, idd in comb:
        comb2 = [(ida, idb, idc, idd), (ida, idc, idb, idd), (ida, idd, idb, idc)]
        random.shuffle(comb2)
        for a, b, c, d in comb2:
            if edge_matrix[a,b]== 1 and edge_matrix[c, d] == 1 and edge_matrix[a, c] == 0 and edge_matrix[a, d] == 0 and edge_matrix[b, c] == 0 and edge_matrix[b, d] == 0:
                ground_edge = [(node, a), (node, c), (node, b), (node, d), (a, node), (b, node), (c, node), (d, node), (a, b), (b, a), (c, d), (d, c)]
                n += 1

                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_67_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 4))
    random.shuffle(combination)
    n = 0
    for ida, idb, idc, idd in combination:
        #combi2 = [(ida,idb, idc,idd), (ida, idc, idd, idb), (ida, idd, idb, idc), (idb, idc, ida, idd), (idb, idd, ida, idc), (idc, idd, ida, idb)]
        combi2 = list(permutations([ida, idb, idc, idd], 4))
        random.shuffle(combi2)
        for a, b, c, d in combi2:
            if edge_matrix[a, c] == 0 and edge_matrix[a, d] == 0 and edge_matrix[a, b] == 1 and edge_matrix[b, d] == 1 and edge_matrix[c, d] == 1 and edge_matrix[b, c] == 1:
                ground_edge = [(node, a), (a, node), (node, b), (b, node), (c, node), (node, c), (node, d), (d, node), (a, b), (b, a),  (b, c), (c, b), (b, d), (d, b), (d, c), (c, d)]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                n+= 1
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if choice_n == n:
                    return max_fider, predict_edge
    if predict_edge == []:
                return 0, []
    return max_fider, predict_edge
def find_8_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    max_fider = -1000
    predict_edge = []
    n = 0
    random.shuffle(neigbor)
    for j in neigbor:
        j_neighbor = list(graph.neighbors(j))
        combination = list(combinations(j_neighbor, 2))
        random.shuffle(combination)
        for node1, node2 in combination:
            if edge_matrix[node1][node2] == 1 and edge_matrix[node1][node] ==0 and edge_matrix[node2][node] == 0:
                n+= 1
                ground_edge = [(node, j), (j, node), (j, node2), (node2, j), (j, node1), (node1, j),
                           (node1, node2), (node2, node1)]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []

    return max_fider, predict_edge


def find_9_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    n = 0
    for a, b in combination:

        if edge_matrix[a][b] == 1:

            a_neighbor = list(graph.neighbors(a))
            a_neighbor.remove(b)
            a_neighbor.remove(node)
            b_neighbor = list(graph.neighbors(b))
            b_neighbor.remove(a)
            b_neighbor.remove(node)
            random.shuffle(a_neighbor)
            random.shuffle(b_neighbor)
            for ida in a_neighbor:
                if edge_matrix[ida, node] == 0 and edge_matrix[ida, b] == 0:
                    n += 1
                    ground_edge = [(node, a), (a, node), (b, node), (node, b), (a, b), (b, a), (ida, a), (a, ida), ]
                    fider = fidelity(model, node, feature, edge, ground_edge, label)
                    if max_fider < fider:
                        predict_edge = ground_edge
                        max_fider = fider
                    if n == choice_n:
                        return max_fider, predict_edge
            for idb in b_neighbor:
                if edge_matrix[idb, node] == 0 and edge_matrix[idb, a] == 0:
                    n += 1
                    ground_edge = [(node, a), (a, node), (b, node), (node, b), (a, b), (b, a), (b, idb), (idb, b)]
                    fider = fidelity(model, node, feature, edge, ground_edge, label)
                    if max_fider < fider:
                        predict_edge = ground_edge
                        max_fider = fider
                    if n == choice_n:
                        return max_fider, predict_edge
    if predict_edge == []:
        print('i dont search 9 orbit')
        return 0, []
    return max_fider, predict_edge


def find_10_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -500
    predict_edge = []
    combination = list(combinations(neigbor, 3))
    random.shuffle(combination)
    n= 0
    for a, b, c in combination:
        abc_list = [(a, b, c), (b, c, a), (c, a, b)]
        random.shuffle(abc_list)
        for ida, idb, idc in abc_list:
            if edge_matrix[ida, idb] == 0 or edge_matrix[idb, idc] == 1 or edge_matrix[idc, ida] == 1:
                continue
            n += 1
            ground_edge = [(ida, node), (node, ida), (node, idb), (idb, node), (idc, node), (node, idc), (ida, idb),
                           (idb, ida)]
            fider = fidelity(model, node, feature, edge, ground_edge, label)
            if max_fider < fider:
                predict_edge = ground_edge
                max_fider = fider
            if n == choice_n:
                return max_fider, predict_edge
    if predict_edge == []:
        return 0, []

    return max_fider, predict_edge
def find_11_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000 ):
    neigbor = list(graph.neighbors(node.item()))
    graph_edge = graph.edges()
    random.shuffle(neigbor)
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    max_fider = -500
    predict_edge = []
    combination2 = []
    n = 0
    for j in combination:
        a, b = j
        if not(a, b) in graph_edge:
            combination2.append((a, b))
    for node1, node2 in combination2:
        node1_neighbor = list(graph.neighbors(node1))
        node2_neighbor = list(graph.neighbors(node2))
        random.shuffle(node1_neighbor)
        random.shuffle(node2_neighbor)
        for j in node1_neighbor:
            if j in node2_neighbor and j != node.item() and j not in neigbor:
                n += 1
                ground_edge = [(node, node1), (node1, node), (node, node2), (node2, node), (j, node1), (node1, j), (j, node2), (node2, j) ]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    max_fider = fider
                    predict_edge = ground_edge
                if choice_n == n:
                    return max_fider, predict_edge
    if predict_edge == []:
        print('i dont search 11 orbit')
        return 0, []
    return max_fider, predict_edge

def find_12_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    n = 0
    predict_edge = []
    random.shuffle(neigbor)
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    max_fider = 0
    for a, b in combination:
        if edge_matrix[a, b] == 0:
            continue
        a_neighbor = list(graph.neighbors(a))
        a_neighbor.remove(node)
        a_neighbor.remove(b)
        b_neighbor = list(graph.neighbors(b))
        b_neighbor.remove(node)
        b_neighbor.remove(a)
        intersec = list(set(a_neighbor)&set(b_neighbor))
        random.shuffle(intersec)
        for c in intersec:
            if edge_matrix[c, node] == 1:
                continue
            ground_edge = [(node, a), (a, node), (b, node), (node, b), (a, b), (b, a), (a, c), (c, a), (c, b), (b, c)]
            fider = fidelity(model, node, feature, edge, ground_edge, label)
            n += 1
            if max_fider < fider:
                predict_edge = ground_edge
                max_fider = fider
            if n == choice_n:
                return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge


def find_15_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    n = 0
    max_fider = 0
    predict_edge = []
    random.shuffle(neigbor)
    for a in neigbor:
        a_neighbor = list(graph.neighbors(a))
        a_neighbor.remove(node)
        random.shuffle(a_neighbor)
        combination = list(combinations(a_neighbor, 3))
        random.shuffle(combination)
        for b, c, d in combination:
            if edge_matrix[node, b] == 0 and edge_matrix[node, c] == 0 and edge_matrix[node, d] == 0 and edge_matrix[
                b, c] == 0 and edge_matrix[b, d] == 0 and edge_matrix[c, d] == 0:
                n += 1
                ground_edge = [(node, a), (a, node), (a, b), (b, a), (c, a), (a, c), (d, a), (a, d)]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_17_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = 0
    predict_edge = []
    node = node.item()
    n = 0
    for fixed_node in neigbor:
        fixed_neighbor = list(graph.neighbors(fixed_node))
        fixed_neighbor.remove(node)
        random.shuffle(fixed_neighbor)
        combination = list(combinations(fixed_neighbor, 2))
        random.shuffle(combination)
        for (a, b) in combination:
            if edge_matrix[a, b] == 0 and edge_matrix[a, node] == 0 and edge_matrix[b, node] == 0:
                neighbor2 = list(graph.neighbors(a))
                random.shuffle(neighbor2)
                neighbor2.remove(fixed_node)
                for l in neighbor2:
                    if l != node and l != b and edge_matrix[l, fixed_node] == 0 and edge_matrix[l, b] == 0 and \
                            edge_matrix[l, node] == 0:
                        n += 1

                        ground_edge = [(fixed_node, node), (node, fixed_node), (fixed_node, a), (a, fixed_node),
                                       (fixed_node, b), (b, fixed_node), (l, a), (a, l)]
                        fider = fidelity(model, node, feature, edge, ground_edge, label)
                        if max_fider < fider:
                            predict_edge = ground_edge
                            max_fider = fider
                        if n == choice_n:
                            return max_fider, predict_edge
                neighbor2 = list(graph.neighbors(b))
                random.shuffle(neighbor2)
                neighbor2.remove(fixed_node)
                for l in neighbor2:
                    if l != node and l != a and edge_matrix[l, fixed_node] == 0 and edge_matrix[l, a] == 0 and \
                            edge_matrix[l, node] == 0:
                        n += 1

                        ground_edge = [(fixed_node, node), (node, fixed_node), (fixed_node, a), (a, fixed_node),
                                       (fixed_node, b), (b, fixed_node), (l, b), (b, l)]
                        fider = fidelity(model, node, feature, edge, ground_edge, label)
                        if max_fider < fider:
                            predict_edge = ground_edge
                            max_fider = fider
                        if n == choice_n:
                            return max_fider, predict_edge

    if predict_edge == []:
        print('i dont search 17 orbit')
        return 0, []
    return max_fider, predict_edge

def find_21_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.choice(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    n = 0
    for a in neigbor:
        a_neighbor = list(graph.neighbors(a))
        random.shuffle(a_neighbor)
        a_neighbor.remove(node)

        combination = list(combinations(a_neighbor, 3))
        random.shuffle(combination)
        for b, c, d in combination:
            if edge_matrix[b, node] == 1 or edge_matrix[c, node] == 1 or edge_matrix[d, node] == 1:
                continue
            abc_list = [(b, c, d), (c, d, b), (d, b, c)]
            random.shuffle(abc_list)
            for idb, idc, idd in abc_list:
                if edge_matrix[idd, idb] == 1 and edge_matrix[idb, idc] == 0 and  edge_matrix[idc, idd] == 0:
                    ground_edge = [(node, a),(a, node), (idb, a), (a, idb), (a, idc), (idc, a), (idd, a), (a, idd), (idd, idb), (idb, idd)]
                    fider = fidelity(model, node, feature, edge, ground_edge, label)
                    n += 1
                    if max_fider < fider:
                        predict_edge = ground_edge
                        max_fider = fider
                    if n == choice_n:
                        return  max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge
def find_22_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    n = 0
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    for a, b in combination:
        if edge_matrix[a][b] == 1:
            ab_list = [(a, b), (b, a)]
            random.shuffle(ab_list)
            for aa, bb in ab_list:
                aa_neighbor = list(graph.neighbors(aa))
                aa_neighbor.remove(node)
                aa_neighbor.remove(bb)
                random.shuffle(aa_neighbor)
                aa_combination = list(combinations(aa_neighbor, 2))
                random.shuffle(aa_combination)
                for ida, idb in aa_combination:

                    if edge_matrix[ida, idb] == 0 and edge_matrix[ida, bb] == 0 and edge_matrix[idb, bb] == 0 and \
                            edge_matrix[ida, node] == 0 and edge_matrix[idb, node] == 0:
                        n+= 1
                        ground_edge = [(node, aa), (aa, node), (bb, node), (node, bb), (aa, bb), (bb, aa), (ida, aa),
                                       (aa, ida), (idb, aa), (aa, idb)]

                        fider = fidelity(model, node, feature, edge, ground_edge, label)
                        if max_fider < fider:
                            predict_edge = ground_edge
                            max_fider = fider
                        if n == choice_n:
                            return max_fider, predict_edge

    if predict_edge == []:
        print('i dont search 22 orbit')
        return 0, []
    return max_fider, predict_edge
def find_27_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    max_fider = 0
    predict_edge = []
    node = node.item()
    n = 0
    random.choice(neigbor)
    for a in neigbor:
        a_neighbor = list(graph.neighbors(a))
        a_neighbor.remove(node)
        random.shuffle(a_neighbor)
        a_combination = list(combinations(a_neighbor, 2))
        random.shuffle(a_combination)
        for b, c in a_combination:
            if edge_matrix[b, c] == 0 or edge_matrix[node, b] == 1 or edge_matrix[c, node] == 1:
                continue
            bc_list = [(b, c), (c, b)]
            random.shuffle(bc_list)
            for idb, idc in bc_list:
                idb_neighbor = list(graph.neighbors(idb))
                random.shuffle(idb_neighbor)
                idb_neighbor.remove(a)
                idb_neighbor.remove(idc)
                if node in idb_neighbor:
                    idb_neighbor.remove(node)
                for d in idb_neighbor:
                    if edge_matrix[node, d] == 0 and edge_matrix[d, a] ==0 and edge_matrix[d, idc] ==0 :
                        ground_edge = [(node,a ), (a, node), (a, idb), (idb, a), (idc, a), (a, idc), (idc, idb), (idb, idc), (d, idb), (idb, d)]
                        fider = fidelity(model, node, feature, edge, ground_edge, label)
                        n += 1
                        if max_fider < fider:
                            predict_edge = ground_edge
                            max_fider = fider
                        if choice_n ==n:
                            return max_fider, predict_edge

    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge
def find_28_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    graph_edge = graph.edges()
    max_fider = -50000
    random.shuffle(neigbor)
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    n = 0
    for a, b in combination:
        if edge_matrix[a][b] == 1:
            a_neighbor = list(graph.neighbors(a))
            random.shuffle(a_neighbor)
            a_neighbor.remove(b)
            a_neighbor.remove(node)
            b_neighbor = list(graph.neighbors(b))
            random.shuffle(b_neighbor)
            b_neighbor.remove(a)
            b_neighbor.remove(node)
            for ida in a_neighbor:
                if edge_matrix[ida,node] == 0 and edge_matrix[ida, b] == 0:
                    for idb in b_neighbor:
                        if edge_matrix[idb, node] == 0 and edge_matrix[idb, a] == 0 and edge_matrix[ida,idb] == 0:
                            n+=1
                            ground_edge = [(node,a), (a, node), (b, node), (node, b), (a, b), (b, a), (ida, a), (a, ida), (b, idb), (idb,b)]
                            fider = fidelity(model, node, feature, edge, ground_edge, label)
                            if max_fider < fider:
                                predict_edge = ground_edge
                                max_fider = fider
                            if n == choice_n:
                                return max_fider, predict_edge
    if predict_edge == []:
        print('i dont search 28 orbit')
        return 0, []

    return max_fider, predict_edge


def find_30_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neighbor = list(graph.neighbors(node.item()))
    n = 0
    random.shuffle(neighbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()

    for a in neighbor:
        a_neighbor = list(graph.neighbors(a))
        random.shuffle(a_neighbor)
        a_neighbor.remove(node)
        a_com = list(combinations(a_neighbor, 2))
        random.shuffle(a_neighbor)
        for b, c in a_com:
            b_neighbor = list(graph.neighbors(b))
            random.shuffle(b_neighbor)
            b_neighbor.remove(a)
            if node in b_neighbor:
                b_neighbor.remove(node)

            c_neighbor = list(graph.neighbors(c))
            random.shuffle(c_neighbor)
            c_neighbor.remove(a)
            if node in c_neighbor:
                c_neighbor.remove(node)
            idd = list(set(b_neighbor) & set(c_neighbor))
            random.shuffle(idd)
            for d in idd:
                if edge_matrix[b, node] == 0 and edge_matrix[c, node] == 0 and edge_matrix[d, node] == 0 and \
                        edge_matrix[a, d] == 0 and edge_matrix[b, c] == 0:

                    ground_edge = [(node, a), (a, node), (b, a), (a, b), (b, d), (d, b), (d, c), (c, d), (a, c), (c, a)]
                    fider = fidelity(model, node, feature, edge, ground_edge, label)
                    n += 1
                    if max_fider < fider:
                        predict_edge = ground_edge
                        max_fider = fider
                    if n == choice_n:
                        return max_fider, predict_edge
    if predict_edge == []:
        return 0, []

    return max_fider, predict_edge

def find_45_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    n = 0
    for ida, idb in combination:
        idcombi = [(ida, idb), (idb, ida)]
        random.shuffle(idcombi)
        for a, b in idcombi:
            if edge_matrix[a, b]== 1:
                continue
            a_neighbor = list(graph.neighbors(a))
            a_neighbor.remove(node)
            a_combi = list(combinations(a_neighbor, 2))
            random.shuffle(a_combi)
            for c, d in a_combi:
                if edge_matrix[c, b] == 1 or edge_matrix[c, node] == 1 or edge_matrix[d, b] == 1 or edge_matrix[d, node] == 1 or edge_matrix[c, d] == 0:
                    continue

                n+=1

                ground_edge = [(node, a),(node, b),(a, node),(b, node) ,(a, c), (c, a), (d, a),
                           (a, d), (c, d), (d, c)]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge


def find_52_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 4))
    random.shuffle(combination)
    n = 0
    for a, b, c, d in combination:
        if edge_matrix[a][b] == 0 and edge_matrix[a, c] ==0 and edge_matrix[a,d ] == 0and edge_matrix[b,c]==1 and edge_matrix[b,d]==1and edge_matrix[c, d]==1:
            n += 1
            ground_edge = [(node, a), (a, node), (node, b), (b, node), (c, node), (node, c), (node, d), (d, node), (b, c), (c, b), (b, d), (d, b), (c,d ), (d, c)]
            fider = fidelity(model, node, feature, edge, ground_edge, label)
            if max_fider < fider:
                predict_edge = ground_edge
                max_fider = fider
        if n == choice_n:
            return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_31_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neighbor = list(graph.neighbors(node.item()))
    random.shuffle(neighbor)
    max_fider = -500
    predict_edge = []
    combination = list(combinations(neighbor, 2))
    random.shuffle(combination)
    n = 0
    for a, b in combination:
        if edge_matrix[a, b] == 0:
            a_neighbor = set(graph.neighbors(a))
            b_neighbor = set(graph.neighbors(b))
            intercept_node =list(a_neighbor & b_neighbor - set(neighbor))
            a_neighbor = list(a_neighbor - set(intercept_node))
            b_neighbor = list(b_neighbor - set(intercept_node))
            random.shuffle(a_neighbor)
            random.shuffle(b_neighbor)
            intercept_node.remove(node.item())
            random.shuffle(intercept_node)
            for ida in a_neighbor:
                if edge_matrix[ida, node] == 0 and edge_matrix[ida, b] == 0:
                    for c in intercept_node:

                        if edge_matrix[c, ida] == 0:
                            n += 1
                            ground_edge = [(node, a), (a, node), (b, node), (node, b), (a, c), (c, a), (b, c), (c, b),
                                           (ida, a), (a, ida)]
                            fider = fidelity(model, node, feature, edge, ground_edge, label)
                            if max_fider < fider:
                                predict_edge = ground_edge
                                max_fider = fider
                            if choice_n == n:
                                return max_fider, predict_edge
            for idb in b_neighbor:
                if edge_matrix[idb, node] == 0 and edge_matrix[idb, a] == 0:
                    for c in intercept_node:
                        if edge_matrix[c, idb] == 0:

                            ground_edge = [(node, a), (a, node), (b, node), (node, b), (a, c), (c, a), (b, c), (c, b),
                                           (idb, b), (b, idb)]

                            fider = fidelity(model, node, feature, edge, ground_edge, label)
                            n += 1
                            if max_fider < fider:
                                predict_edge = ground_edge
                                max_fider = fider
                            if n == choice_n:
                                return max_fider, predict_edge

    if predict_edge == []:
        print('i dont search 31 orbit')
        return 0, []

    return max_fider, predict_edge
def find_33_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    n = 0
    combination = list(combinations(neigbor, 3))
    random.shuffle(combination)
    max_fider = 0
    predict_edge = []
    for a, b, c in combination:
        if edge_matrix[a, b] == 1 or edge_matrix[b, c] == 1 or edge_matrix[c, a] == 1:
            continue
        abc_list = [(a, b, c), (b, c, a), (c, a, b)]
        random.shuffle(abc_list)
        for ida, idb, idc in abc_list:
            idb_neighbor = list(graph.neighbors(idb))
            idb_neighbor.remove(node)
            idc_neighbor = list(graph.neighbors(idc))
            idc_neighbor.remove(node)
            intersec_node = list(set(idb_neighbor)&set(idc_neighbor))
            random.shuffle(intersec_node)
            for d in intersec_node:
                if edge_matrix[d, node] == 0 and edge_matrix[d, ida] ==0:
                    n += 1
                    ground_edge = [(node, ida), (ida, node), (node, idb), (idb, node), (idc, node), (node, idc),
                                   (idb, d), (d, idb), (idc, d), (d, idc)]
                    fider = fidelity(model, node, feature, edge, ground_edge, label)
                    if max_fider < fider:
                        predict_edge = ground_edge
                        max_fider = fider
                    if choice_n == n:
                        return max_fider, predict_edge

    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge


def find_14_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 3))
    random.shuffle(combination)
    n = 0
    for a, b, c in combination:
        if edge_matrix[a, b] == 1 and edge_matrix[b, c] == 1 and edge_matrix[c, a] == 1:

            ground_edge = [(node, a), (a, node), (node, b), (b, node), (c, node), (node, c), (a, b),
                           (b, a), (b, c), (c, b), (a, c), (c, a)]
            fider = fidelity(model, node, feature, edge, ground_edge, label)
            n += 1
            if max_fider < fider:
                predict_edge = ground_edge
                max_fider = fider
            if choice_n == n:
                return max_fider, predict_edge
    if predict_edge == []:
        return 0, []

    return max_fider, predict_edge


def find_22_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    n = 0
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    for a, b in combination:
        if edge_matrix[a][b] == 1:
            a_neighbor = list(graph.neighbors(a))
            a_neighbor.remove(node)
            a_neighbor.remove(b)
            a_combination = list(combinations(a_neighbor, 2))
            b_neighbor = list(graph.neighbors(b))
            b_neighbor.remove(node)
            b_neighbor.remove(a)
            b_combination = list(combinations(b_neighbor, 2))
            random.shuffle(a_combination)
            random.shuffle(b_combination)
            for ida1, ida2 in a_combination:

                if edge_matrix[ida1, ida2] == 0 and edge_matrix[ida1, b] == 0 and edge_matrix[ida2, b] == 0 and \
                        edge_matrix[ida1, node] == 0 and edge_matrix[ida2, node] == 0:
                    n += 1
                    ground_edge = [(node, a), (a, node), (b, node), (node, b), (a, b), (b, a), (ida1, a), (a, ida1),
                                   (a, ida2), (ida2, a)]
                    fider = fidelity(model, node, feature, edge, ground_edge, label)
                    if max_fider < fider:
                        predict_edge = ground_edge
                        max_fider = fider
                    if n == choice_n:
                        return max_fider, predict_edge
            for idb1, idb2 in b_combination:

                if edge_matrix[idb1, idb2] == 0 and edge_matrix[idb1, a] == 0 and edge_matrix[idb2, a] == 0 and \
                        edge_matrix[idb1, node] == 0 and edge_matrix[idb2, node] == 0:
                    n += 1
                    ground_edge = [(node, a), (a, node), (b, node), (node, b), (a, b), (b, a), (idb1, b), (b, idb1),
                                   (b, idb2), (idb2, b)]
                    fider = fidelity(model, node, feature, edge, ground_edge, label)
                    if max_fider < fider:
                        predict_edge = ground_edge
                        max_fider = fider
                    if choice_n == n:
                        return max_fider, predict_edge
    if predict_edge == []:
        print('i dont search 22 orbit')
        return 0, []

    return max_fider, predict_edge

def find_19_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    max_fider = -500
    node = node.item()
    predict_edge = []
    n = 0

    for a_, b_ in combination:
        if edge_matrix[a_][b_] == 0:
            for a, b in [(a_, b_), (b_, a_)]:
                a_neighbor = list(graph.neighbors(a))
                random.shuffle(a_neighbor)
                a_neighbor.remove(node)
                if b in a_neighbor:
                    a_neighbor.remove(b)
                combination2 = list(combinations(a_neighbor, 2))
                random.shuffle(combination2)
                for ida1, ida2 in combination2:
                    if edge_matrix[ida1][ida2] == 0 and edge_matrix[ida1][node] ==0 and edge_matrix[ida2][node] ==0 and edge_matrix[ida1][b] ==0 and edge_matrix[ida2][b] ==0:
                        n += 1

                        ground_edge = [(b, node), (node, b), (node, a), (a, node), (ida1, a), (a, ida1), (a, ida2), (ida2, a)]
                        fider = fidelity(model, node, feature, edge, ground_edge, label)
                        if max_fider < fider:
                            predict_edge = ground_edge
                            max_fider = fider
                        if choice_n == n:
                            return max_fider, predict_edge
    if predict_edge == []:
        print('i dont search 19 orbit')
        return 0, []
    return max_fider, predict_edge

def find_26_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    n = 0
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    for a_, b_ in combination:
        if edge_matrix[a_][b_] == 0:
            for a, b in [(a_, b_), (b_, a_)]:
                a_neighbor = list(graph.neighbors(a))
                a_neighbor.remove(node)
                random.shuffle(a_neighbor)
                if b in a_neighbor:
                    a_neighbor.remove(b)
                for c in a_neighbor:
                    if edge_matrix[c][node] == 0 and edge_matrix[c][b] == 0:
                        c_neighbor = list(graph.neighbors(c))
                        random.shuffle(c_neighbor)

                        if node in c_neighbor:
                            c_neighbor.remove(node)
                        if b in c_neighbor:
                            c_neighbor.remove(b)
                        for d in c_neighbor:
                            if edge_matrix[d][a] == 0 and edge_matrix[d][node] == 0 and edge_matrix[d][b] == 0:
                                n += 1
                                ground_edge = [(node, a), (a, node), (b, node), (node, b), (a, c), (c, a), (c, d),
                                               (d, c)]
                                fider = fidelity(model, node, feature, edge, ground_edge, label)
                                if max_fider < fider:
                                    predict_edge = ground_edge
                                    max_fider = fider
                                if n == choice_n:
                                    return max_fider, predict_edge

    if predict_edge == []:
        print('i dont search 26 orbit')
        return 0, []
    return max_fider, predict_edge

def find_46_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neighbor = list(graph.neighbors(node.item()))
    n = 0
    max_fider = -50000
    predict_edge = []
    node = node.item()
    random.shuffle(neighbor)
    for a in neighbor:
        a_neighbor = list(graph.neighbors(a))
        random.shuffle(a_neighbor)
        a_neighbor.remove(node)
        a_com = list(combinations(a_neighbor, 2))
        random.shuffle(a_com)
        for b, c in a_com:
            b_neighbor = list(graph.neighbors(b))
            random.shuffle(b_neighbor)
            b_neighbor.remove(a)
            if node in b_neighbor:
                b_neighbor.remove(node)
            c_neighbor = list(graph.neighbors(c))
            random.shuffle(c_neighbor)
            c_neighbor.remove(a)
            if node in c_neighbor:
                c_neighbor.remove(node)
            idd = list(set(b_neighbor) & set(c_neighbor))
            random.shuffle(idd)
            for d in idd:
                if edge_matrix[b, node] == 0 and edge_matrix[c, node] == 0 and edge_matrix[d, node] == 0 and \
                        edge_matrix[a, d] == 0 and edge_matrix[b, c] == 1:

                    ground_edge = [(node, a), (a, node), (b, a), (a, b), (b, d), (d, b), (d, c), (c, d), (a, c), (c, a), (b, c), (c, b)]
                    fider = fidelity(model, node, feature, edge, ground_edge, label)
                    n += 1
                    if max_fider < fider:
                        predict_edge = ground_edge
                        max_fider = fider
                    if n == choice_n:
                        return max_fider, predict_edge

    if predict_edge == []:
        return 0, []

    return max_fider, predict_edge

def find_32_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neighbor = list(graph.neighbors(node.item()))
    random.shuffle(neighbor)
    max_fider = 0
    predict_edge = []
    combination = list(combinations(neighbor, 2))
    random.shuffle(combination)
    n = 0
    for a, b in combination:
        if edge_matrix[a,b] == 0:
            a_neighbor = list(graph.neighbors(a))
            random.shuffle(a_neighbor)
            a_neighbor = set(a_neighbor)
            b_neighbor = list(graph.neighbors(b))
            random.shuffle(b_neighbor)
            b_neighbor = set(b_neighbor)
            intercept_node = list(a_neighbor&b_neighbor)
            random.shuffle(intercept_node)
            intercept_node.remove(node.item())
            for c in intercept_node:
                if edge_matrix[c, node] == 1:
                    continue
                c_neighbor = list(graph.neighbors(c))
                random.shuffle(c_neighbor)
                c_neighbor = set(c_neighbor)
                c_neighbor.remove(a)
                c_neighbor.remove(b)
                if node in c_neighbor:
                    c_neighbor.remove(node)
                for d in c_neighbor:
                    if edge_matrix[d, node] == 1 or edge_matrix[d, a]== 1 or edge_matrix[d, b] == 1:
                        continue

                    ground_edge= [(node, a), (a, node), (node, b), (b, node), (a, c), (c, a), (b, c), (c, b), (d, c), (c, d)]

                    fider = fidelity(model, node, feature, edge, ground_edge, label)
                    n += 1
                    if max_fider < fider:
                        predict_edge = ground_edge
                        max_fider = fider
                    if n == choice_n:
                        return max_fider, predict_edge
    if predict_edge == []:
        return 0, []

    return max_fider, predict_edge

def find_29_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    n = 0
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 3))
    random.shuffle(combination)

    for a_, b_, c_ in combination:
        abc_list = [(a_, b_, c_), (b_, c_, a_), (c_, a_, b_)]
        for a, b, c in abc_list:
            if edge_matrix[a][b] == 1 and edge_matrix[a][c] == 0 and edge_matrix[b][c] == 0:
                a_neighbor = list(graph.neighbors(a))
                a_neighbor.remove(node)
                a_neighbor.remove(b)
                b_neighbor = list(graph.neighbors(b))
                b_neighbor.remove(node)
                b_neighbor.remove(a)
                random.shuffle(a_neighbor)
                random.shuffle(b_neighbor)
                for ida in a_neighbor:

                    if ida != c and edge_matrix[ida, node] == 0 and edge_matrix[ida, c] == 0 and edge_matrix[ida, b] == 0:
                        ground_edge = [(node, c), (c, node), (node, a), (a, node), (b, node), (node, b), (ida, a),
                                       (a, ida), (a, b), (b, a)]
                        n += 1
                        fider = fidelity(model, node, feature, edge, ground_edge, label)
                        if max_fider < fider:
                            predict_edge = ground_edge
                            max_fider = fider
                        if n == choice_n:
                            return max_fider, predict_edge
                for idb in b_neighbor:

                    if idb != c and edge_matrix[idb, node] == 0 and edge_matrix[idb, c] == 0 and edge_matrix[idb, a] == 0:
                        ground_edge = [(node, c), (c, node), (node, a), (a, node), (b, node), (node, b), (idb, b),
                                       (b, idb), (a, b), (b, a)]
                        n += 1
                        fider = fidelity(model, node, feature, edge, ground_edge, label)
                        if max_fider < fider:
                            predict_edge = ground_edge
                            max_fider = fider
                        if n == choice_n:
                            return max_fider, predict_edge

    if predict_edge == []:
        print('i dont search 29 orbit')
        return 0, []
    return max_fider, predict_edge
def find_23_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 4))
    random.shuffle(combination)
    n = 0
    for a, b, c, d in combination:
        combination_list = [(a, b, c, d), (b, c, d, a), (c, d, a, b), (d, a, b, c), (a, c, d, b), (b, d, a, c)]
        random.shuffle(combination_list)
        for ida, idb, idc, idd in combination_list:
            if edge_matrix[ida, idb] == 1 and edge_matrix[idb, idc] == 0 and edge_matrix[idc, idd] == 0 and edge_matrix[idd, ida] == 0 and edge_matrix[ida, idc] == 0 and edge_matrix[idb, idd] == 0:
                ground_edge = [(ida, idb), (idb, ida), (node, ida),(ida, node), (node, idb), (idb, node), (idc, node), (node, idc), (idd, node), (node, idd) ]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                n += 1
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_44_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    #print(neigbor)
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 3))
    random.shuffle(combination)
    n = 0
    for ida, idb, idc in combination:
        combi = [(ida, idb, idc), (idb, idc, ida), (idc, ida, idb)]
        random.shuffle(combi)
        for a, b, c in combi:
            if edge_matrix[b, c] == 0 or edge_matrix[a, b]== 1or edge_matrix[a,c ] == 1:
                continue

            a_neighbor = list(graph.neighbors(a))
            random.shuffle(a_neighbor)
            a_neighbor.remove(node)
            for d in a_neighbor:
                #print(d)
                if edge_matrix[node, d] == 0 and edge_matrix[d, c] == 0 and edge_matrix[d,b ]== 0:

                    ground_edge = [(node, a), (node, b), (node, c), (a, node), (b, node), (c, node), (a, d), (d, a),(b, c), (c, b)]
                    n+= 1
                    fider = fidelity(model, node, feature, edge, ground_edge, label)
                    if max_fider < fider:
                        predict_edge = ground_edge
                        max_fider = fider
                    if n == choice_n:
                        return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge
def find_55_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    n= 0
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    for a, b in combination:
        if edge_matrix[a][b] == 0:
            a_neighbor =  list(graph.neighbors(a))
            b_neighbor = list(graph.neighbors(b))
            a_neighbor.remove(node)
            b_neighbor.remove(node)
            random.shuffle(a_neighbor)
            random.shuffle(b_neighbor)
            for c in a_neighbor:
                if edge_matrix[node, c] == 0 and edge_matrix[b, c] == 0:
                    for d in b_neighbor:
                        if edge_matrix[d, node] == 0 and edge_matrix[d, a] ==0 and edge_matrix[c, d] == 1:
                            ground_edge = [(node, a), (a, node), (b, node), (node, b), (a, c), (c, a), (b, d), (d, b), (c, d), (d, c)]
                            fider = fidelity(model, node, feature, edge, ground_edge, label)
                            n+= 1
                            if max_fider < fider:
                                predict_edge = ground_edge
                                max_fider = fider
                            if n == choice_n:
                                return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge
def find_43_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    n = 0
    for a, b in combination:
        if edge_matrix[a][b] == 1:
            ab_list = [(a, b), (b, a)]
            random.shuffle(ab_list)
            for a1, b1 in ab_list:
                a1_neighbor = list(graph.neighbors(a1))
                a1_neighbor.remove(node)
                a1_neighbor.remove(b1)
                random.shuffle(a1_neighbor)
                for a2 in a1_neighbor:
                    if edge_matrix[a2, node] == 0 and edge_matrix[a2, b1] == 0:
                        a2_neighbor = list(graph.neighbors(a2))
                        a2_neighbor.remove(a1)
                        random.shuffle(a2_neighbor)
                        if b1 in a2_neighbor:
                            a2_neighbor.remove(b1)
                        if node in a2_neighbor:
                            a2_neighbor.remove(node)
                        for a3 in a2_neighbor:
                            if edge_matrix[a3, a1] == 0 and edge_matrix[a3, b1] == 0 and edge_matrix[a3, node] == 0:
                                n += 1
                                ground_edge = [(node, a1), (a1, node), (b1, node), (node, b1), (a1, a2), (a2, a1),
                                               (a3, a2), (a2, a3)]
                                fider = fidelity(model, node, feature, edge, ground_edge, label)
                                if max_fider < fider:
                                    predict_edge = ground_edge
                                    max_fider = fider
                                if n == choice_n:
                                    return max_fider, predict_edge

    if predict_edge == []:
        print('i dont search 43 orbit')
        return 0, []
    return max_fider, predict_edge
def find_27_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    n= 0
    max_fider = 0
    predict_edge = []
    node = node.item()
    for a in neigbor:
        a_neighbor = list(graph.neighbors(a))
        a_neighbor.remove(node)
        random.shuffle(a_neighbor)
        a_combination = list(combinations(a_neighbor, 2))
        random.shuffle(a_combination)
        for b, c in a_combination:
            if edge_matrix[b, c] == 0 or edge_matrix[node, b] == 1 or edge_matrix[c, node] == 1:
                continue
            bc_list = [(b, c), (c, b)]
            random.shuffle(bc_list)
            for idb, idc in bc_list:
                idb_neighbor = list(graph.neighbors(idb))
                idb_neighbor.remove(a)
                idb_neighbor.remove(idc)
                random.shuffle(idb_neighbor)
                if node in idb_neighbor:
                    idb_neighbor.remove(node)
                for d in idb_neighbor:
                    if edge_matrix[node, d] == 0 and edge_matrix[d, a] ==0 and edge_matrix[d, idc] ==0 :
                        n += 1
                        ground_edge = [(node,a ), (a, node), (a, idb), (idb, a), (idc, a), (a, idc), (idc, idb), (idb, idc), (d, idb), (idb, d)]
                        fider = fidelity(model, node, feature, edge, ground_edge, label)
                        if max_fider < fider:
                            predict_edge = ground_edge
                            max_fider = fider
                        if n == choice_n:
                            return max_fider, predict_edge
    if predict_edge == []:
        print('idont find 27 orbit')
        return 0, []
    return max_fider, predict_edge
def find_69_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 4))
    random.shuffle(combination)
    n = 0
    for i in combination:
        combination2 = list(combinations(i, 2))
        random.shuffle(combination2)
        combination3 = list(combinations(combination2, 2))
        random.shuffle(combination3)
        for j in combination3:

            if len(set(j[0])-set(j[1]))<2:
                continue
            ground_edge = []
            a, b = j
            if edge_matrix[a[0], a[1]] ==0 and edge_matrix[b[0], b[1]] == 0:
                for l in combination2:
                    if not(l in j) and edge_matrix[l[0], l[1]] == 1:
                        ground_edge.extend([(l[0], l[1]), (l[1], l[0])])
                if len(ground_edge) == 8:
                    ground_edge.extend([(i[0], node), (i[1], node),(i[2], node),(i[3], node), (node, i[0]), (node, i[1]), (node, i[2]),(node, i[3])])
                    fider = fidelity(model, node, feature, edge, ground_edge, label)
                    n += 1
                    if max_fider < fider:
                        predict_edge = ground_edge
                        max_fider = fider
                    if n == choice_n:
                        return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_36_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 3))
    random.shuffle(combination)
    n = 0
    for ida, idb, idc in combination:
        idcombi = [(ida, idb, idc),(idb, ida, idc), (idc, idb, ida) ]
        random.shuffle(idcombi)
        for a, b, c in idcombi:
            if edge_matrix[a, b]== 0 or edge_matrix[a, c]== 0 or edge_matrix[c, b]== 1:
                continue
            a_neighbor = list(graph.neighbors(a))
            a_neighbor.remove(node)
            a_neighbor.remove(b)
            a_neighbor.remove(c)
            random.shuffle(a_neighbor)
            for d in a_neighbor:
                if edge_matrix[b,d] == 1 or edge_matrix[c, d] == 1 or edge_matrix[node, d] == 1:
                    continue

                n+=1

                ground_edge = [(node, a),(node, b),(node, c),(a, node),(b, node) , (c, node),(a, b), (b, a), (a, c),
                               (c, a), (a, d), (d, a)]

                fider = fidelity(model, node, feature, edge, ground_edge, label)

                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_13_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    n = 0
    random.shuffle(neigbor)
    graph_edge = graph.edges()
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    max_fider = 0
    predict_edge = []
    combination2 = []
    for j in combination:
        a, b = j
        if not(a, b) in graph_edge:
            combination2.append((a, b))
    random.shuffle(combination2)
    for node1, node2 in combination2:
        node1_neighbor = list(graph.neighbors(node1))
        node2_neighbor = list(graph.neighbors(node2))
        random.shuffle(node1_neighbor)
        random.shuffle(node2_neighbor)
        for j in node1_neighbor:
            if j in node2_neighbor and j != node.item() and edge_matrix[j, node] == 1:
                ground_edge = [(node, node1), (node1, node), (node, node2), (node2, node), (j, node1), (node1, j), (j, node2), (node2, j) ]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                n+=1
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []

def find_70_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 3))
    n = 0
    for a, b, c in combination:
        if edge_matrix[a, b]== 1 and edge_matrix[a, c] == 1 and edge_matrix[b, c] == 1:
            a_neighbor = set(graph.neighbors(a))
            b_neighbor = set(graph.neighbors(b))
            c_neighbor = set(graph.neighbors(c))
            inter = list(a_neighbor&b_neighbor&c_neighbor)
            inter.remove(node)
            for d in inter:
                if edge_matrix[d, node] == 0:
                    n += 1
                    ground_edge = [(node, a), (node, b),(node, c),(a, node), (b, node),(c, node),(a, b), (b, a), (b, c), (c, b), (a, d), (d, a), (d, b), (b,d), (c, d), (d, c) ]
                    fider = fidelity(model, node, feature, edge, ground_edge, label)

                    if max_fider < fider:
                        predict_edge = ground_edge
                        max_fider = fider
                    if n == choice_n:
                        return max_fider, predict_edge

    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_65_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    max_fider = 0
    predict_edge = []
    n = 0
    for a, b in combination:
        if edge_matrix[a, b] == 0:
            continue
        a_neighbor = list(graph.neighbors(a))
        a_neighbor.remove(node)
        a_neighbor.remove(b)
        random.shuffle(a_neighbor)
        a_combination = list(combinations(a_neighbor, 2))
        random.shuffle(a_combination)
        for ida, idb in a_combination:
            if edge_matrix[node, ida] == 0 and edge_matrix[node, idb] ==0 and  edge_matrix[ida, idb]==1 and edge_matrix[ida, b]== 1 and edge_matrix[idb, b] == 1:
                n += 1
                ground_edge = [(node, a), (a, node), (node, b), (b, node), (a, b), (b, a),
                               (ida, a), (a, ida), (ida, b), (b, ida), (b, idb), (idb, b), (a, idb), (idb, a), (idb, ida), (ida, idb)]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_37_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    max_fider = -50000
    predict_edge = []
    node = node.item()
    random.shuffle(neigbor)
    combination = list(combinations(neigbor, 4))
    random.shuffle(combination)
    n = 0
    for ida, idb, idc, idd in combination:
        idcombi = [(ida, idb, idc, idd),(idb, ida, idc, idd), (ida, idc, idb, idd),(idc, ida, idb, idd), (ida, idd, idb, idc), (idd, ida, idb, idc), (idb, idc, ida, idd),(idc, idb, ida, idd), (idb, idd, ida, idc), (idd, idb, ida, idc), (idc, idd, ida, idb),(idd, idc, ida, idb) ]
        random.shuffle(idcombi)
        for a, b, c, d in idcombi:
            if edge_matrix[a, c]== 1 or edge_matrix[a, b]== 1 or edge_matrix[a, d]== 1 or edge_matrix[b, c] == 0 or edge_matrix[b, d] ==0 or edge_matrix[c, d] == 1:
                continue


            n+=1

            ground_edge = [(node, a),(node, b),(node, c), (node, d), (a, node),(b, node) , (c, node),(d, node),(c, b), (b, c), (b, d),
                           (d, b)]

            fider = fidelity(model, node, feature, edge, ground_edge, label)

            if max_fider < fider:
                predict_edge = ground_edge
                max_fider = fider
            if n == choice_n:
                return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_48_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 3))
    random.shuffle(combination)
    n = 0
    for ida, idb, idc in combination:
        idcombi = [(ida, idb, idc), (ida, idc, idb), (idb, idc, ida), (idb, ida, idc), (idc, idb,ida), (idc, ida, idb)]
        random.shuffle(idcombi)
        for a, b, c in idcombi:
            if edge_matrix[a, b]== 0 or edge_matrix[c, b]== 0 or edge_matrix[a, c]== 1:
                continue
            c_neighbor = list(graph.neighbors(c))
            c_neighbor.remove(node)

            c_neighbor.remove(b)
            for d in c_neighbor:
                if edge_matrix[node, d] == 1 or edge_matrix[b, d] == 1 or edge_matrix[a, d] == 1:
                    continue
                n+=1

                ground_edge = [(node, a),(node, b),(node, c), (a, node),(b, node) , (c, node),(a, b), (b, a), (b, c),
                           (c, b), (c, d), (d, c)]

                fider = fidelity(model, node, feature, edge, ground_edge, label)

                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_60_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 3))
    random.shuffle(combination)
    n = 0
    for ida, idb, idc in combination:
        idcombi = [(ida, idb, idc), (ida, idc, idb), (idb, idc, ida), (idb, ida, idc), (idc, idb,ida), (idc, ida, idb)]
        random.shuffle(idcombi)
        for a, b, c in idcombi:
            if edge_matrix[a, b]== 0 or edge_matrix[c, b]== 0 or edge_matrix[a, c]== 1:
                continue
            c_neighbor = list(graph.neighbors(c))
            c_neighbor.remove(node)
            c_neighbor.remove(b)
            b_neighbor = list(graph.neighbors(b))
            b_neighbor.remove(node)
            b_neighbor.remove(c)
            b_neighbor.remove(a)
            intersect = list(set(b_neighbor)&set(c_neighbor))
            random.shuffle(intersect)
            for d in intersect:
                if edge_matrix[node, d] == 1 or edge_matrix[a, d] == 1:
                    continue
                n+=1

                ground_edge = [(node, a),(node, b),(node, c), (a, node),(b, node) , (c, node),(a, b), (b, a), (b, c),
                           (c, b), (c, d), (d, c), (b, d), (d, b)]

                fider = fidelity(model, node, feature, edge, ground_edge, label)

                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge


def find_61_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 4))
    random.shuffle(combination)
    n = 0
    for ida, idb, idc, idd in combination:
        #combi2 = [(ida, idb, idc, idd), (ida, idc, idd, idb), (ida, idd, idb, idc), (ida, idb, idd, idc), (ida, idc, idb, idd), (ida, idd, idc, idb)]
        combi2 = list(permutations([ida, idb,idc,idd], 4))
        random.shuffle(combi2)
        for a, b, c, d in combi2:
            if edge_matrix[a, c] == 1 and edge_matrix[c, d] == 1 and edge_matrix[b, d] == 1 and edge_matrix[a, d] == 0 and edge_matrix[c, b] ==0 and edge_matrix[a,b]==0:
                n+=1
                ground_edge = [(node, a), (node, b), (node, c), (node, d),(a, node),(b, node),(c, node),(d, node),(a, c), (c, d), (c, a), (d, c), (b, d), (d, b)]

                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if choice_n == n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_71_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    n = 0
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 4))
    random.shuffle(combination)
    for a, b, c, d in combination:
        abcd_list = [(a, b, c, d), (b, c, d, a), (c, d, a, b), (d, a, b, c), (a, c, b, d), (b, d, c, a)]
        random.shuffle(abcd_list)
        for ia, ib, ic, id in abcd_list:
            if edge_matrix[ia, ib] == 0 and edge_matrix[ia, ic] == 1 and edge_matrix[ia, id] == 1 and edge_matrix[ib, id] == 1 and edge_matrix[ib, ic] == 1 and edge_matrix[ic, id] == 1:
                ground_edge = [(node, ia), (ia, node), (ib, node), (node, ib), (node, ic), (ic, node), (ic, ia), (ia, ic), (ic, ib), (ib, ic), (node, id), (id, node), (id, ib),
                               (ib, id), (ic, id), (id, ic), (id, ia), (ia, id)]
                n += 1
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []

    return max_fider, predict_edge
def find_72_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    n = 0

    for a, b in combination:
        if edge_matrix[a][b] == 1:
            a_neighbor = list(graph.neighbors(a))
            a_neighbor.remove(node)
            a_neighbor.remove(b)
            b_neighbor = list(graph.neighbors(b))
            b_neighbor.remove(node)
            b_neighbor.remove(a)
            random.shuffle(a_neighbor)
            random.shuffle(b_neighbor)
            for ida in a_neighbor:
                if edge_matrix[ida,node] == 1 and edge_matrix[ida, b] == 1:
                    for idb in b_neighbor:
                        if ida> idb and edge_matrix[idb, node] == 1 and edge_matrix[idb, a] == 1 and edge_matrix[ida,idb] == 1:
                            n += 1
                            ground_edge = [(node,a), (a, node), (b, node), (node, b), (a, b), (b, a), (ida, a), (a, ida), (b, idb), (idb,b), (ida, node),
                                           (node, ida), (b, ida), (ida, b), (idb, node), (node, idb), (idb, a), (a, idb), (idb, ida), (ida, idb)]
                            fider = fidelity(model, node, feature, edge, ground_edge, label)
                            if max_fider < fider:
                                predict_edge = ground_edge
                                max_fider = fider
                            if n == choice_n:
                                return max_fider, predict_edge
    if predict_edge == []:
        return 0, []

    return max_fider, predict_edge
def find_39_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 3))
    random.shuffle(combination)
    n= 0
    for a, b, c in combination:
        if edge_matrix[a, b] == 0 and edge_matrix[b, c] == 0 and edge_matrix[c, a] == 0:
            a_neighbor = list(graph.neighbors(a))
            a_neighbor.remove(node)
            b_neighbor = list(graph.neighbors(b))
            b_neighbor.remove(node)
            c_neighbor = list(graph.neighbors(c))
            c_neighbor.remove(node)
            a_neighbor, b_neighbor, c_neighbor =  set(a_neighbor), set(b_neighbor), set(c_neighbor)
            corss_node = list(a_neighbor&b_neighbor&c_neighbor)
            random.shuffle(corss_node)
            for d in corss_node:
                if edge_matrix[d, node] ==0:
                    n += 1
                    ground_edge = [(node, a), (a, node), (b, node), (node, b), (c, node), (node, c),(d, a), (a, d), (b, d), (d, b), (c, d), (d, c)]
                    fider = fidelity(model, node, feature, edge, ground_edge, label)
                    if max_fider < fider:
                        predict_edge = ground_edge
                        max_fider = fider
                    if n == choice_n:
                        return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_34_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    n = 0
    for a in neigbor:
        a_neighbor = list(graph.neighbors(a))
        a_neighbor.remove(node)
        random.shuffle(a_neighbor)
        combination = list(combinations(a_neighbor, 3))
        random.shuffle(combination)
        for idb, idc, idd in combination:
            if edge_matrix[node, idb] == 0 and edge_matrix[node, idc ] == 0 and edge_matrix[node, idd] == 0:
                combi_list = [(idb, idc, idd),(idc, idb, idd),(idd, idc, idb)]
                random.shuffle(combi_list)
                for b, c, d in combi_list:
                    if edge_matrix[c, b] == 1 and edge_matrix[b, d] == 1 and edge_matrix[c, d] == 0:
                        n += 1
                        ground_edge = [(node, a), (a, node), (a, c), (c, a), (a, b), (b, a),(a, d), (d, a),(b, c), (c, b), (b, d), (d, b)]
                        fider = fidelity(model, node, feature, edge, ground_edge, label)
                        if max_fider < fider:
                            predict_edge = ground_edge
                            max_fider = fider
                        if n == choice_n:
                            return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_41_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    n = 0
    combi = list(combinations(neigbor, 4))
    random.shuffle(combi)
    for ida, idb, idc, idd in combi:
        abcd = [(ida, idb, idc, idd), (idb, idc, idd, ida), (idc, idd, ida, idb), (idd, ida, idb, idc)]
        random.shuffle(abcd)
        for a, b, c, d in abcd:
            if edge_matrix[a, b] == 1 and edge_matrix[a, c] == 1 and edge_matrix[a, d] == 1 and edge_matrix[b, c] == 0 and edge_matrix[b, d] ==0 and edge_matrix[c, d] ==0:
                n += 1
                ground_edge = [(node, a), (node, b), (node, c), (node, d),(a, node),(b, node), (c, node), (d, node), (a, b), (b, a),(c, a), (a, c),(d, a), (a, d) ]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_68_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combi_ = list(combinations(neigbor, 3))
    random.shuffle(combi_)
    n = 0
    for ida, idb, idc in combi_:
        combination = [(ida, idb, idc), (idb, idc, ida), (idc, idb, ida)]
        random.shuffle(combination)
        for a, b, c in combination:
            if edge_matrix[a, b] == 1 and edge_matrix[a, c] == 1 and edge_matrix[c, b] == 0:
                a_neighbor = list(graph.neighbors(a))
                a_neighbor.remove(node)
                b_neighbor = list(graph.neighbors(b))
                b_neighbor.remove(node)
                c_neighbor = list(graph.neighbors(c))
                c_neighbor.remove(node)
                a_neighbor, b_neighbor, c_neighbor =  set(a_neighbor), set(b_neighbor), set(c_neighbor)
                corss_node = list(a_neighbor&b_neighbor&c_neighbor)
                random.shuffle(corss_node)
                for d in corss_node:
                    if edge_matrix[d, node] ==0:
                        n += 1
                        ground_edge = [(node, a), (a, node), (b, node), (node, b), (c, node), (node, c),(d, a), (a, d), (b, d), (d, b), (c, d), (d, c)]
                        fider = fidelity(model, node, feature, edge, ground_edge, label)
                        if max_fider < fider:
                            predict_edge = ground_edge
                            max_fider = fider
                        if n == choice_n:
                            return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge
def find_20_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 3))
    random.shuffle(combination)
    n = 0
    for ida, idb, idc in combination:
        if edge_matrix[ida, idb] == 0 and edge_matrix[idb, idc] == 0 and edge_matrix[idc, ida] == 0:
            combi = [(ida, idb, idc), (idb, idc, ida), (idc, ida, idb)]
            random.shuffle(combi)
            for a, b, c in combi:
                a_neighbor = list(graph.neighbors(a))
                a_neighbor.remove(node)
                random.shuffle(a_neighbor)
                for d in a_neighbor:
                    if edge_matrix[node, d] == 0 and edge_matrix[d, c] == 0 and edge_matrix[d,b ]== 0:
                        ground_edge = [(node, a), (node, b), (node, c), (a, node), (b, node), (c, node), (a, d), (d, a)]

                        fider = fidelity(model, node, feature, edge, ground_edge, label)
                        n += 1
                        if max_fider < fider:
                            predict_edge = ground_edge
                            max_fider = fider
                        if n == choice_n:
                            return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_18_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    n = 0
    for a in neigbor:
        a_neighbor = list(graph.neighbors(a))
        a_neighbor.remove(node)
        random.shuffle(a_neighbor)
        for b in a_neighbor:
            if edge_matrix[node, b] == 1:
                continue

            b_neighbor = list(graph.neighbors(b))
            random.shuffle(b_neighbor)
            b_neighbor.remove(a)
            if node in b_neighbor:
                b_neighbor.remove(node)
            combination = list(combinations(b_neighbor, 2))
            random.shuffle(combination)
            for c, d in combination:
                if edge_matrix[node, c] == 1 or edge_matrix[node, d] == 1 or edge_matrix[a, c] == 1 or edge_matrix[a, d] == 1 or edge_matrix[c, d] == 1:
                    continue
                n += 1
                ground_edge = [(node, a), (a, node),(a, b), (b, a), (c, b), (b, c), (d, b), (b, d)]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_42_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    n = 0
    for a in neigbor:
        a_neighbor = list(graph.neighbors(a))
        a_neighbor.remove(node)
        random.shuffle(a_neighbor)
        for b in a_neighbor:
            if edge_matrix[node, b] == 1:
                continue
            b_neighbor = list(graph.neighbors(b))
            b_neighbor.remove(a)
            random.shuffle(b_neighbor)
            if node in b_neighbor:
                b_neighbor.remove(node)
            combination = list(combinations(b_neighbor, 2))
            random.shuffle(combination)
            for c, d in combination:
                if edge_matrix[node, c] == 1 or edge_matrix[node, d] == 1 or edge_matrix[a, c] == 1 or edge_matrix[a, d] == 1 or edge_matrix[c, d] == 0:
                    continue
                n += 1
                ground_edge = [(node, a), (a, node),(a, b), (b, a), (c, b), (b, c), (d, b), (b, d), (c, d), (d, c)]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_38_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    n =0
    for a, b in combination:
        if edge_matrix[a, b] == 1:
            continue
        a_neighbor = list(graph.neighbors(a))
        b_neighbor = list(graph.neighbors(b))
        random.shuffle(a_neighbor)
        random.shuffle(b_neighbor)
        a_neighbor.remove(node)
        if b in a_neighbor:
            a_neighbor.remove(b)
        if a in b_neighbor:
            b_neighbor.remove(a)
        b_neighbor.remove(node)
        section = list(set(a_neighbor)&set(b_neighbor))
        random.shuffle(section)
        combi2 = list(combinations(section, 2))
        random.shuffle(combi2)

        for c, d in combi2:
            if edge_matrix[node, c] == 1 or edge_matrix[node, d] == 1 or edge_matrix[c, d]==1:
                continue
            n +=1
            ground_edge = [(node, a), (a, node), (b, node), (node, b), (a, c), (c, a), (b, c), (c, b), (a, d), (d, a), (b, d), (d, b)]
            fider = fidelity(model, node, feature, edge, ground_edge, label)
            if max_fider < fider:
                predict_edge = ground_edge
                max_fider = fider
            if n == choice_n:
                return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_40_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    n = 0
    for a, b in combination:
        if edge_matrix[a, b] == 0:
            continue
        a_neighbor = list(graph.neighbors(a))
        b_neighbor = list(graph.neighbors(b))
        a_neighbor.remove(node)
        random.shuffle(a_neighbor)
        random.shuffle(b_neighbor)
        if b in a_neighbor:
            a_neighbor.remove(b)
        if a in b_neighbor:
            b_neighbor.remove(a)
        b_neighbor.remove(node)
        section = list(set(a_neighbor)&set(b_neighbor))
        random.shuffle(section)
        combi2 = list(combinations(section, 2))
        random.shuffle(combi2)
        for c, d in combi2:
            if edge_matrix[node, c] == 1 or edge_matrix[node, d] == 1 or edge_matrix[c, d]==1:
                continue
            n += 1
            ground_edge = [(node, a), (a, node), (b, node), (node, b), (a, c), (c, a), (b, c), (c, b), (a, d), (d, a), (b, d), (d, b), (a, b), (b, a)]
            fider = fidelity(model, node, feature, edge, ground_edge, label)
            if max_fider < fider:
                predict_edge = ground_edge
                max_fider = fider
            if n == choice_n:
                return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_63_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 3))
    random.shuffle(combination)
    n = 0
    for ida, idb, idc in combination:
        idcombi = [(ida, idb, idc),(idb, ida, idc), (idc, idb, ida) ]
        random.shuffle(idcombi)
        for a, b, c in idcombi:
            if edge_matrix[a, b]== 0 or edge_matrix[a, c]== 0 or edge_matrix[c, b]== 1:
                continue
            b_neighbor = list(graph.neighbors(b))
            c_neighbor = list(graph.neighbors(c))
            b_neighbor.remove(a)
            b_neighbor.remove(node)
            c_neighbor.remove(a)
            c_neighbor.remove(node)
            intersect = list(set(b_neighbor)&set(c_neighbor))
            random.shuffle(intersect)
            for d in intersect:
                if edge_matrix[a, d] == 1 or edge_matrix[node,d] == 1:
                    continue
                n+=1

                ground_edge = [(node, a),(node, b),(node, c),(a, node),(b, node) , (c, node),(a, b), (b, a), (a, c),
                               (c, a), (b, d), (d, b), (c, d), (d, c)]

                fider = fidelity(model, node, feature, edge, ground_edge, label)

                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge
def find_64_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 3))
    random.shuffle(combination)
    n = 0
    for ida, idb, idc in combination:
        idcombi = [(ida, idb, idc),(idb, ida, idc), (idc, idb, ida) ]
        random.shuffle(idcombi)
        for a, b, c in idcombi:
            if edge_matrix[a, b]== 1 or edge_matrix[a, c]== 1 or edge_matrix[c, b]== 0:
                continue
            b_neighbor = list(graph.neighbors(b))
            c_neighbor = list(graph.neighbors(c))

            b_neighbor.remove(node)

            c_neighbor.remove(node)
            intersect = list(set(b_neighbor)&set(c_neighbor))
            random.shuffle(intersect)
            for d in intersect:
                if edge_matrix[a, d] == 0 or edge_matrix[node,d] == 1:
                    continue
                n+=1

                ground_edge = [(node, a),(node, b),(node, c),(a, node),(b, node) , (b,c), (c, b), (a, d),(d, a),(b, d), (d, b), (c, d), (d, c) ]

                fider = fidelity(model, node, feature, edge, ground_edge, label)

                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge


def find_50_orbit(graph, node, label, model, feature, edge, edge_matrix,  choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    n = 0
    for a in neigbor:
        a_neighbor = list(graph.neighbors(a))
        a_neighbor.remove(node)
        a_combination = list(combinations(a_neighbor, 3))
        random.shuffle(a_combination)
        for b, c, d in a_combination:
            if edge_matrix[node, b] == 1 or edge_matrix[node, c] == 1 or edge_matrix[node, d] == 1 or edge_matrix[b, c] == 0 or edge_matrix[c, d] == 0 or edge_matrix[b, d] == 0:
                continue
            n +=1
            ground_edge = [(node, a), (a, node), (a, b), (b, a), (a, c), (c, a), (a, d), (d, a), (b, d), (d, b), (b, c),
                           (c, b), (c, d), (d, c)]
            fider = fidelity(model, node, feature, edge, ground_edge, label)
            if max_fider < fider:
                predict_edge = ground_edge
                max_fider = fider
            if n == choice_n:
                return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_51_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):

    neighbor = list(graph.neighbors(node.item()))
    random.shuffle(neighbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neighbor, 3))
    random.shuffle(combination)
    n = 0
    for ida, idb, idc in combination:
        if edge_matrix[ida, idb] == 0 or edge_matrix[ida, idc] == 0 or edge_matrix[idb, idc] == 0:
            continue
        combi = [(ida, idb, idc), (idb, ida, idc), (idc, ida, idb)]
        random.shuffle(combi)
        for a, b, c in combi:
            a_neighbor = list(graph.neighbors(a))
            a_neighbor.remove(b)
            a_neighbor.remove(c)
            a_neighbor.remove(node)
            random.shuffle(a_neighbor)
            for d in a_neighbor:
                if edge_matrix[d, b] == 1 or edge_matrix[d, c] == 1 or edge_matrix[d, node] == 1:
                    continue
                ground_edge = [(node, a), (a, node), (node, b), (b, node), (node, c), (c, node), (a, c), (c, a), (a, b),
                               (b, a), (b, c), (c, b), (a, d), (d, a)]
                n += 1

                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge


def find_66_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 3))
    random.shuffle(combination)
    n = 0
    for ida, idb, idc in combination:
        if edge_matrix[ida, idb] == 0 or edge_matrix[ida, idc] == 0 or edge_matrix[idc, idb] == 0:
            continue
        combi2 = [(ida,idb, idc), (idb, idc, ida), (idc, ida, idb)]
        random.shuffle(combi2)
        for a, b, c in combi2:
            a_neighbor = list(graph.neighbors(a))
            b_neighbor = list(graph.neighbors(b))
            a_neighbor.remove(node)
            a_neighbor.remove(b)
            a_neighbor.remove(c)
            b_neighbor.remove(node)
            b_neighbor.remove(a)
            b_neighbor.remove(c)
            cross = list(set(a_neighbor)&set(b_neighbor))
            random.shuffle(cross)
            for d in cross:
                if edge_matrix[d, node] == 1 or edge_matrix[d, c] == 1:
                    continue
                n += 1
                ground_edge = [(node, a), (a, node), (node, b), (b, node), (c, node), (node, c), (a, b), (b, a), (a, c), (c, a), (b, c), (c, b), (a, d), (d, a), (d, b), (b, d)]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
                return 0, []
    return max_fider, predict_edge

def find_25_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    n = 0
    for a, b in combination:
        if edge_matrix[a][b] == 0:
            a_neighbor =  list(graph.neighbors(a))
            b_neighbor = list(graph.neighbors(b))
            a_neighbor.remove(node)
            b_neighbor.remove(node)
            random.shuffle(a_neighbor)
            random.shuffle(b_neighbor)
            for c in a_neighbor:
                if edge_matrix[node, c] == 0 and edge_matrix[b, c] == 0:
                    for d in b_neighbor:
                        if edge_matrix[d, node] == 0 and edge_matrix[d, a] ==0 and edge_matrix[c, d] == 0:
                            n += 1


                            ground_edge = [(node, a), (a, node), (b, node), (node, b), (a, c), (c, a), (b, d), (d, b),]
                            fider = fidelity(model, node, feature, edge, ground_edge, label)
                            if max_fider < fider:
                                predict_edge = ground_edge
                                max_fider = fider
                            if n == choice_n:
                                return max_fider, predict_edge

    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_62_orbit(graph, node, label, model, feature, edge, edge_matrix,  choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    #print(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    n = 0
    for a, b in combination:
        if edge_matrix[a][b] == 1:
            continue
        a_neighbor = list(graph.neighbors(a))
        a_neighbor.remove(node)
        b_neighbor = list(graph.neighbors(b))
        b_neighbor.remove(node)
        intersect_node = list(set(a_neighbor)&set(b_neighbor))
        intersect_combi = list(combinations(intersect_node, 2))
        random.shuffle(intersect_combi)
        for c, d in intersect_combi:
            if edge_matrix[c][d] == 0 or edge_matrix[c][node] == 1 or edge_matrix[d][node] == 1:
                continue
            n += 1

            ground_edge = [(node, a), (a, node), (b, node), (node, b), (c, d), (d, c), (a, c), (c, a), (b, c), (c, b), (a, d), (d, a),(d, b), (b, d)]
            fider = fidelity(model, node, feature, edge, ground_edge, label)
            if max_fider < fider:
                predict_edge = ground_edge
                max_fider = fider
            if n == choice_n:
                return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_53_orbit(graph, node, label, model, feature, edge, edge_matrix,  choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    comb = list(combinations(neigbor, 2))
    random.shuffle(comb)
    n = 0
    for ida, idb in comb:
        comb2 = [(ida, idb), (idb, ida)]
        random.shuffle(comb2)
        for a, b in comb2:
            if edge_matrix[a,b]== 0:
                continue
            b_neighbor = list(graph.neighbors(b))
            b_neighbor.remove(a)
            b_neighbor.remove(node)
            b_combi = list(combinations(b_neighbor, 2))
            random.shuffle(b_combi)
            for c, d in b_combi:
                if edge_matrix[c, d] == 0 or edge_matrix[c, node] == 1 or edge_matrix[c, a] == 1 or edge_matrix[d, node] == 1 or edge_matrix[d, a] == 1:
                    continue
                ground_edge = [(node, a), (node, b), (a, node), (b, node), (a, b), (b, a), (b, c), (c, b), (b, d), (d, b),(c, d), (d, c)]
                n += 1

                fider = fidelity(model, node, feature, edge, ground_edge, label)

                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider

                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_56_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    max_fider = -50000
    predict_edge = []
    node = node.item()
    comb = list(combinations(neigbor, 2))
    random.shuffle(comb)
    n = 0
    for a, b in comb:
        if edge_matrix[a, b] == 0:
            continue
        a_neighbor = list(graph.neighbors(a))
        b_neighbor = list(graph.neighbors(b))
        a_neighbor.remove(node)
        a_neighbor.remove(b)
        b_neighbor.remove(node)
        b_neighbor.remove(a)
        random.shuffle(a_neighbor)
        random.shuffle(b_neighbor)
        for c in a_neighbor:
            for d in b_neighbor:
                if c == d or edge_matrix[c, d] == 0 or edge_matrix[c, b] == 1 or edge_matrix[c, node] == 1 or edge_matrix[a, d] == 1 or edge_matrix[d, node] == 1:
                    continue
                ground_edge = [(node, a), (a, node), (b, node), (node, b), (a, b), (b, a), (a, c), (c, a), (c, d), (d, c), (b, d), (d, b)]
                n += 1
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_59_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    max_fider = -50000
    predict_edge = []
    node = node.item()
    comb = list(combinations(neigbor, 2))
    random.shuffle(comb)
    n = 0
    for ida, idb in comb:
        if edge_matrix[ida, idb] == 0:
            continue
        comb2 = [(ida, idb), (idb, ida)]
        random.shuffle(comb2)
        for a, b in comb2:
            a_neighbor = list(graph.neighbors(a))
            a_neighbor.remove(b)
            a_neighbor.remove(node)
            a_combi = list(combinations(a_neighbor, 2))
            random.shuffle(a_combi)
            for idc, idd in a_combi:
                comb3 = [(idc, idd), (idd, idc)]
                random.shuffle(comb3)
                for c, d in comb3:
                    if edge_matrix[c, d] == 0 or edge_matrix[c, b] == 1 or edge_matrix[b, d] == 0 or edge_matrix[node, c] == 1 or edge_matrix[node, d] == 1:
                        continue
                    ground_edge = [(node, a), (a, node), (b, node), (node, b), (a, b), (b, a), (a, d), (d, a), (b, d), (d, b), (a, c), (c, a), (c, d), (d, c)]
                    n += 1

                    fider = fidelity(model, node, feature, edge, ground_edge, label)
                    if max_fider < fider:
                        predict_edge = ground_edge
                        max_fider = fider
                    if n == choice_n:
                        return max_fider, predict_edge
    if predict_edge == []:
        return 0, []
    return max_fider, predict_edge

def find_57_orbit(graph, node, label, model, feature, edge, edge_matrix,  choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    n = 0

    for ida, idb in combination:
        if edge_matrix[ida, idb] == 1:
            continue
        combi2 = [(ida,idb), (idb,  ida)]
        random.shuffle(combi2)
        for a, b in combi2:
            a_neighbor = list(graph.neighbors(a))
            a_neighbor.remove(node)
            random.shuffle(a_neighbor)
            combi3 = list(combinations(a_neighbor, 2))

            for idc, idd in combi3:
                if edge_matrix[idc, idd] == 0 or edge_matrix[node, idc] == 1 or edge_matrix[node, idd] == 1:
                    continue
                combi4 = [(idc, idd), (idd, idc)]
                random.shuffle(combi4)
                for c, d in combi4:
                    if edge_matrix[c, b] == 1 or edge_matrix[d, b] == 0:
                        continue
                    n += 1

                    ground_edge = [(node, a), (a, node), (node, b),(b, node), (b, d), (d, b), (a, d), (d, a), (a, c), (c, a), (c, d), (d, c)]
                    fider = fidelity(model, node, feature, edge, ground_edge, label)
                    if max_fider < fider:
                        predict_edge = ground_edge
                        max_fider = fider
                    if n == choice_n:
                        return max_fider, predict_edge

    if predict_edge == []:
                return 0, []
    return max_fider, predict_edge


def find_24_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    random.shuffle(neigbor)
    max_fider = -50000
    predict_edge = []
    node = node.item()
    n = 0
    for a in neigbor:
        a_neigbor = list(graph.neighbors(a))
        a_neigbor.remove(node)
        random.shuffle(a_neigbor)
        for b in a_neigbor:
            if edge_matrix[node, b] == 1:
                continue
            b_neigbor = list(graph.neighbors(b))
            b_neigbor.remove(a)
            random.shuffle(b_neigbor)
            for c in b_neigbor:
                if edge_matrix[c, node] == 1 or edge_matrix[c, a] == 1:
                    continue
                c_neigbor = list(graph.neighbors(c))
                c_neigbor.remove(b)
                random.shuffle(c_neigbor)
                for d in c_neigbor:
                    if edge_matrix[d, node] == 1 or edge_matrix[d, a] == 1 or edge_matrix[d, b] == 1:
                        continue
                    ground_edge = [(node, a), (a, node), (a, b), (b, a), (c, b), (b, c), (d, c), (c, d)]
                    fider = fidelity(model, node, feature, edge, ground_edge, label)
                    #fider = 0
                    n+= 1

                    if max_fider < fider:
                        predict_edge = ground_edge
                        max_fider = fider
                    if n == choice_n:
                        return max_fider, predict_edge
    if predict_edge == []:
                return 0, []
    return max_fider, predict_edge

def find_35_orbit(graph, node, label, model, feature, edge, edge_matrix,choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 2))
    n = 0
    random.shuffle(combination)

    for ida, idb in combination:
        if edge_matrix[ida, idb] == 0:
            continue
        combi2 = [(ida,idb), (idb,  ida)]
        random.shuffle(combi2)
        for a, b in combi2:
            b_neighbor = list(graph.neighbors(b))
            b_neighbor.remove(a)
            b_neighbor.remove(node)
            combi3 = list(combinations(b_neighbor, 2))
            random.shuffle(combi3)

            for idc, idd in combi3:
                if edge_matrix[idc, idd] == 1 :
                    continue
                combi4 = [(idc, idd), (idd, idc)]
                random.shuffle(combi4)
                for c, d in combi4:
                    if edge_matrix[node, c] == 1 or edge_matrix[node, d] == 1 or edge_matrix[a, c] == 0 or edge_matrix[a, d] == 1:
                        continue
                    n += 1

                    ground_edge = [(node, a), (a, node), (node, b),(b, node), (b, d), (d, b), (a, b), (b, a), (a, c), (c, a), (b,c),(c,b)]
                    fider = fidelity(model, node, feature, edge, ground_edge, label)
                    if max_fider < fider:
                        predict_edge = ground_edge
                        max_fider = fider
                    if n == choice_n:
                        return max_fider, predict_edge

    if predict_edge == []:
                return 0, []
    return max_fider, predict_edge

def find_47_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 2))
    random.shuffle(combination)
    n = 0

    for a, b in combination:
        if edge_matrix[a, b] == 0:
            continue
        a_neighbor = list(graph.neighbors(a))
        a_neighbor.remove(node)
        random.shuffle(a_neighbor)
        for c in a_neighbor:
            if edge_matrix[c, b] == 0 or edge_matrix[c, node] == 1:
                continue
            c_neighbor  = list(graph.neighbors(c))
            c_neighbor.remove(a)
            c_neighbor.remove(b)
            random.shuffle(c_neighbor)
            for d in c_neighbor:
                if edge_matrix[node, d] == 1 or edge_matrix[a, d] == 1 or edge_matrix[b, d]:
                    continue
                n += 1

                ground_edge = [(node, a), (a, node), (node, b),(b, node), (a, b), (b, a), (a, c), (c, a), (b, c), (c, b), (c, d), (d, c)]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge

    if predict_edge == []:
                return 0, []
    return max_fider, predict_edge

def find_49_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 3))
    random.shuffle(combination)
    n = 0

    for ida, idb, idc in combination:
        combi2 = [(ida, idb, idc), (ida, idc, idb), (idb, idc, ida)]
        random.shuffle(combi2)
        for a, b, c in combi2:
            if edge_matrix[a, b] == 0 or edge_matrix[a, c] == 1 or edge_matrix[b, c] == 1:
                continue
            a_neighbor = list(graph.neighbors(a))
            a_neighbor.remove(b)
            a_neighbor.remove(node)
            random.shuffle(a_neighbor)
            for d in a_neighbor:
                if edge_matrix[d, b] == 0 or edge_matrix[d, node] == 1 or edge_matrix[d, c] == 1:
                    continue
                n += 1

                ground_edge = [(node, a), (a, node), (node, b),(b, node), (a, b), (b, a), (a, d), (d, a), (b, d), (d, b), (node, c), (c, node)]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge

    if predict_edge == []:
                return 0, []
    return max_fider, predict_edge

def find_58_orbit(graph, node, label, model, feature, edge, edge_matrix, choice_n = 1000):
    neigbor = list(graph.neighbors(node.item()))
    max_fider = -50000
    predict_edge = []
    node = node.item()
    combination = list(combinations(neigbor, 3))
    random.shuffle(combination)
    n = 0

    for ida, idb, idc in combination:
        combi2 = [(ida, idb, idc), (ida, idc, idb), (idb, idc, ida), (idb, ida, idc), (idc, ida, idb), (idc, idb, ida)]
        random.shuffle(combi2)
        for a, b, c in combi2:
            if edge_matrix[a, c] == 0 or edge_matrix[a, b] == 1 or edge_matrix[b, c] == 1:
                continue
            b_neighbor = list(graph.neighbors(b))
            b_neighbor.remove(node)
            random.shuffle(b_neighbor)
            for d in b_neighbor:
                if edge_matrix[d, c] == 0 or edge_matrix[d, node] == 1 or edge_matrix[d, a] == 1:
                    continue
                n += 1

                ground_edge = [(node, a), (a, node), (node, b),(b, node), (node, c), (c, node), (a, c), (c, a), (b, d), (d, b), (c,d) , (d, c)]
                fider = fidelity(model, node, feature, edge, ground_edge, label)
                if max_fider < fider:
                    predict_edge = ground_edge
                    max_fider = fider
                if n == choice_n:
                    return max_fider, predict_edge

    if predict_edge == []:
                return 0, []
    return max_fider, predict_edge