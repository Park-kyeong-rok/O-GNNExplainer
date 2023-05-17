#775번째 주석 풀어라
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from train_model.model import  *
from arguments import args
from torchviz import make_dot
from collections import deque
import copy
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from sklearn.model_selection import train_test_split
#from evaluation import Plot_orbit_similarity_map, equi_acc_remove
import torch_geometric
from orbit_find import *
from torch_geometric.utils import k_hop_subgraph

def load_model():
    data_path = f'data/{args.data}/'
    model_path = f'data/{args.data}/model/{args.model_name}'
    if args.agg:
        model_path = f'data/{args.data}/agg_model/{args.model_name}'
    print(model_path)
    model_dict = torch.load(model_path)

    if args.model_name.startswith('ppi'):


        model = Net(50, 1, 100)
        model.load_state_dict(model_dict)

        return model
    else:
        hidden_state = list(model_dict.values())[0].shape[0]
        label = list(model_dict.values())[-2].shape[0]
        input_dim = list(model_dict.values())[1].shape[1]

        if args.agg:
            model = eval(f'agg_GCN{(len(model_dict)-5)//3}({input_dim}, {label}, {hidden_state})')
        else:
            if args.num_model_layer == 2:
                model = eval(f'GCN{2}({input_dim}, {label}, {hidden_state})')

            elif  args.num_model_layer == 3:
                model = eval(f'GCN{3}({input_dim}, {label}, {hidden_state})')
    #model = Netsimple(10, 10)
    model.load_state_dict(model_dict)

    return model

def load_raw_model():
    data_path = f'data/{args.data}/'
    model_path = f'data/{args.data}/model/{args.model_name}'
    if args.agg:
        model_path = f'data/{args.data}/agg_model/{args.model_name}'
    model_dict = torch.load(model_path)

    hidden_state =list(model_dict.values())[0].shape[0]
    label = list(model_dict.values())[-1].shape[0]
    input_dim = list(model_dict.values())[1].shape[1]
    if args.agg:
        model = eval(f'agg_GCN{(len(model_dict)-5)//3}({input_dim}, {label}, {hidden_state})')
    else:
        model = eval(f'GCN{(len(model_dict) - 5) // 3}({input_dim}, {label}, {hidden_state})')


    return model

def load_data(data):
    data_path = f'data/{data}/'
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

def load_orbit(num_node=700, reg = False, answer = [0]):
    data_path = f'data/{args.data}/orbit.txt'
    raw_orbit = np.genfromtxt(f'{data_path}', dtype=np.int32)
    orbit = np.zeros((num_node, 73))

    for i in raw_orbit:
        orbit[i[0],:] = i[1:]
    raw_orbit = copy.copy(orbit)
    orbit = orbit[answer]


    n_answer = torch.sum(answer, dim=0).item()
    exist_orbit = np.sum((orbit>0).astype(int), axis=0)

    not_using_orbit = np.where(exist_orbit == n_answer)[0].tolist()
    not_using_orbit.extend(np.where(exist_orbit == 0)[0].tolist())
    same_dict = Plot_orbit_similarity_map(orbit, not_using_orbit)
    not_using_orbit.append(0)
    for v in same_dict.values():
        not_using_orbit.extend(v)
    not_using_orbit = list(set(not_using_orbit))
    if args.num_model_layer < 4:
        not_using_orbit.append(24)
        not_using_orbit.append(42)
    if args.num_model_layer < 3:
        not_using_orbit.extend([6, 8, 17, 18, 21, 26, 27, 30, 32, 34, 43, 45, 46, 47, 50, 53, 55, 56, 57, 59, 62, 65])


    use_orbit = [i for i in range(73) if i not in not_using_orbit]
    print(f'---------we use orbit(total: {len(use_orbit)})------------')
    print(f'{use_orbit}')

    if reg == False:
        return (orbit > 0).astype(int)[:,use_orbit], use_orbit, same_dict
    else:
        return orbit[:,use_orbit], use_orbit, same_dict


def Plot_orbit_similarity_map(orbit, not_using_orbit, reg=args.reg):
    same_dict = dict()
    values_ox = set()
    if not(reg):
        orbit = (orbit>0).astype(np.int)

    for o1 in range(73):
        same_list = []
        if o1 in not_using_orbit:
            continue
        for o2 in range(o1+1, 73):
            if o2 in not_using_orbit:
                continue
            similarity = np.sum(orbit[:, o1] == orbit[:, o2])/orbit.shape[0]
            print(o1, o2, similarity)
            #if similarity == 1 or similarity >= 0.99:
            if similarity == 1 or similarity >= 0.99:
            #if similarity == 1:
                same_list.append(o2)
        values_ox = values_ox|set(same_list)
        if o1 not in values_ox and same_list != []:
            same_dict[o1] = same_list
    print('---------simirality info-----------')
    for k, v in same_dict.items():
        print(f'{k}: {v}')
    print('-----------------------------------')



    return same_dict
def result(train_answer_list, train_pred_list):
    train_roc = roc_auc_score(train_answer_list, train_pred_list)
    fpr, tpr, thr = roc_curve(train_answer_list, train_pred_list)
    train_optimal_idx = np.argmax(tpr - fpr)
    train_optimal_threshold = thr[train_optimal_idx]
    train_acc = (torch.FloatTensor(train_pred_list)>train_optimal_threshold).float()
    train_acc = torch.mean((train_acc==torch.FloatTensor(train_answer_list)).float())
    return train_roc, train_acc
def select_the_learnable_orbit():
    data_path = f'data/{args.data}/orbit.txt'
    raw_orbit = np.genfromtxt(f'{data_path}', dtype=np.int32)[:, 1:]
    learnable_orbit = []



    for i in range(73):
        orbit_label = (raw_orbit[:,i]>0).astype(int)
        train_raw_model(orbit_label, i,learnable_orbit)
    print(learnable_orbit)


def train_raw_model(orbit_label, orbit_number, learnable_orbit):
    model= load_raw_model()
    feature,_,edge = load_data(args.data)
    lr = 0.002
    epochs = 3000
    orbit_label = torch.LongTensor(orbit_label)
    data = torch_geometric.data.Data(x=feature, edge_index=edge, y=orbit_label)
    node_num = data.x.shape[0]
    input_dim = data.x.shape[1]
    label = data.y.numpy()

    label_num = len(np.unique(label))
    train_mask, test_mask, train_label, test_label = train_test_split([i for i in range(node_num)], label,
                                                                      test_size=0.1, stratify=label, random_state=25)
    train_mask, validation_mask, _, _ = train_test_split(train_mask, train_label, test_size=1 / 9, stratify=train_label,
                                                         random_state=25)

    loss_f = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_list = []
    val_loss_list = []
    val_acc = 0
    val_loss = 100000
    last_train_acc = 0
    for epoch in range(1, epochs + 1):
        result = model(x=data.x, edge_index=data.edge_index)
        loss = loss_f(result[train_mask], data.y[train_mask])
        loss_list.append(loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = torch.nn.functional.log_softmax(result, dim=1)
        answer = torch.argmax(result, dim=1)
        answer = answer == data.y
        validation_loss = loss_f(result[validation_mask], data.y[validation_mask])
        val_loss_list.append(validation_loss.detach().numpy())
        train_acc = torch.mean(answer[train_mask].float())
        validation_acc = torch.mean(answer[validation_mask].float())

        if epoch >= 100 and validation_loss <= val_loss:
            val_loss = validation_loss
            val_acc = validation_acc
            last_train_acc = train_acc
            torch.save(model.state_dict(), f'data/{args.data}/just_kidding/aa')
    plt.plot(loss_list)
    plt.plot(val_loss_list)
    plt.legend(['train', 'validation'])
    plt.title(f'orbit_number: {orbit_number}')
    plt.show()
    model.load_state_dict(torch.load(f'data/{args.data}/just_kidding/aa'))
    final = model(x=data.x, edge_index=data.edge_index)
    final = torch.nn.functional.softmax(final, dim=1)
    final = torch.argmax(final, dim=1)
    score = final == data.y

    val_acc = torch.mean(score[validation_mask].float()).item()
    train_acc = torch.mean(score[train_mask].float()).item()
    test_acc = torch.mean(score[test_mask].float()).item()
    total_list = len(test_mask) + len(train_mask) + len(validation_mask)
    totol_acc = val_acc * len(validation_mask) / total_list + train_acc * len(train_mask) / total_list + test_acc * len(
        test_mask) / total_list
    if totol_acc>= 0.9:
        learnable_orbit.append(orbit_number)
    result_txt = open(
        f'data/{args.data}/just_kidding/{args.agg}_{10}.txt','a')
    result_txt.write(f'{orbit_number}\ntotal_acc: {totol_acc}, train_acc: {train_acc}, val_acc: {val_acc}, test_acc: {test_acc}\n')



class binary_classification:
    def __init__(self, use_orbit, representation, y):
        self.y = y
        self.use_orbit = use_orbit
        #representation = answer_node x hidden
        self.representation = representation
        #Linear = hi
        self.binary_classifier = [torch.nn.Linear(representation.shape[1], 1, bias=args.bias) for _ in
                                  range(len(use_orbit))]
        self.orbit_weight = torch.zeros((representation.shape[1], len(use_orbit)))

        zero_idx = torch.unsqueeze(torch.squeeze(torch.argwhere((torch.sum(representation, dim=0) == 0).long())),dim=0)


        for lin in self.binary_classifier:
            lin_weight = lin.weight.detach()

            lin_weight[:,zero_idx] = 0
            lin.weight = torch.nn.Parameter(lin_weight)

            torch.nn.init.xavier_uniform_(lin.weight)






    def Dataset_and_Dataloader(self, x, y, batch):
        # 0 대입시 배치 없이 학습이 진행
        class dataset(Dataset):
            def __init__(self):
                self.x = x
                self.y = y

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                x = self.x[idx]
                y = torch.unsqueeze(self.y[idx], dim = -1)
                return x, y

        data_set = dataset()
        if batch == 0:
            batch = len(data_set)
        return DataLoader(data_set, batch_size=batch, shuffle=True, drop_last=False)


    def train(self, orbit_idx, orbit, reg= args.reg):
        device = f'cuda:{args.cuda_n}' if torch.cuda.is_available() else 'cpu'
        model = self.binary_classifier[orbit_idx]
        model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr = args.b_lr)
        if reg:
            loss_f = torch.nn.MSELoss()
        else:
            loss_f = torch.nn.BCEWithLogitsLoss()
        if args.focal_loss:
            loss_f = FocalLoss()
        if args.OHEM_loss:
            loss_f = ohemloss_f()

        dataloader = self.Dataset_and_Dataloader(x= self.representation,y= self.y[:, orbit_idx], batch=args.b_batch)
        max_acc = 0
        max_roc = 0
        answer_list = []
        min_loss = 100000
        batch_len = len(dataloader)
        loss_list = []
        acc_list = []

        for epoch in range(1, args.b_epoch+1):
            epoch_loss = 0
            a_acc = 0
            pred_list = []
            answer_list = []
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                predict = model(x)

                if args.focal_loss:
                    predict = torch.sigmoid(predict)

                loss = loss_f(predict, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if not(reg):
                    acc = torch.mean(((torch.sigmoid(predict)>0.5).long() == y).float())

                    if args.focal_loss:
                        acc = torch.mean(((predict > 0.5).long() == y).float())
                with torch.no_grad():
                    epoch_loss += loss.detach()
                    predict = torch.squeeze(predict)
                    y = torch.squeeze(y)
                    pred_list.extend(torch.sigmoid(predict).detach().tolist())
                    answer_list.extend(y.tolist())
                a_acc += acc
            epoch_acc, roc = result(answer_list, pred_list)
            epoch_loss /= batch_len
            a_acc /= batch_len
            # if epoch_acc >= max_acc and not(reg) and epoch_loss <= min_loss:
            #     optimized_weight = model.weight.detach()
            #     max_acc = epoch_acc
            #     min_loss = epoch_loss
            if roc >= max_roc and epoch_loss <= min_loss:
                optimized_weight = model.weight.cpu().detach()
                max_acc = epoch_acc
                max_roc = roc
                min_loss = epoch_loss
            # else:
            #     if epoch_loss < min_loss:
            #         optimized_weight = model.weight.detach()
            #         min_loss = epoch_loss
            loss_list.append(epoch_loss.to('cpu'))
            acc_list.append(epoch_acc)
            #if epoch_acc == 1 and epoch>=100:
            if epoch_acc == 1:
                optimized_weight = model.weight.cpu().detach()
                max_acc = epoch_acc
                max_roc = roc
                min_loss = epoch_loss
                print(f'epoch: {epoch}, loss: {epoch_loss}, acc: {epoch_acc}, roc: {roc}')
                break
            if epoch%100 == 0:
                print(f'epoch: {epoch}, loss: {epoch_loss}, acc: {epoch_acc}, roc: {roc}')
        plt.plot([i for i in range(1, len(loss_list)+1)], loss_list)
        plt.plot([i for i in range(1, len(loss_list)+1)], acc_list)
        plt.title(f'{orbit} weight learning')
        plt.show()
        return optimized_weight, max_acc

    def run(self, orbit_acc, reg=args.reg):
        #orbit_weight.shape = hidden x used_orbit

        orbit_weight = self.orbit_weight
        not_use_orbit = []
        survived_orbit_idx = [i for i in range(orbit_weight.shape[1])]
        orbit_weight_info = []
        new_orbit_acc = []
        for orbit_idx, orbit in enumerate(self.use_orbit):
            print(f'{orbit}s weight learning start')
            #optimized_weight = 1 x hidden
            optimized_weight, max_acc = self.train(orbit_idx, orbit)
            if not(reg):
                if max_acc <0.99 and orbit_acc[orbit_idx] -0.005< max_acc <orbit_acc[orbit_idx] +0.005:
                    not_use_orbit.append((orbit_idx, orbit))
                #풀기
                elif max_acc < 0.90:
                    not_use_orbit.append((orbit_idx, orbit))
                elif max_acc < orbit_acc[orbit_idx]:
                    not_use_orbit.append((orbit_idx, orbit))
                else:
                    orbit_weight_info.append(max_acc)
                    new_orbit_acc.append(orbit_acc[orbit_idx])

            orbit_weight[:, orbit_idx] = optimized_weight

        self.new_orbit_acc = new_orbit_acc
        for orbit_idx, orbit in not_use_orbit:
            self.use_orbit.remove(orbit)
            survived_orbit_idx.remove(orbit_idx)
        self.orbit_weight = orbit_weight[:,survived_orbit_idx]
        self.normalize()
        if not(reg):
            self.orbit_weight_info = orbit_weight_info
        self.save()


    def normalize(self):
        orbit_weight = self.orbit_weight
        orbit_weight = orbit_weight/(torch.sum(orbit_weight**2,dim=0)**(1/2))
        orbit_weight_regular = self.orbit_weight-torch.mean(self.orbit_weight, dim=0)
        orbit_weight_regular = orbit_weight_regular / (torch.sum(orbit_weight_regular ** 2, dim=0) ** (1 / 2))
        self.orbit_weight = orbit_weight
        self.orbit_weight_regular = orbit_weight_regular

    def save(self):
        orbit_weight_list = [self.use_orbit, self.orbit_weight, self.orbit_weight_regular]
        if args.agg:
            file_path = f'result/{args.data}/agg_model/{args.model_name}/{args.b_lr}_{args.b_batch}_{args.b_epoch}{args.subforder}'
        else:
            file_path = f'result/{args.data}/model/{args.model_name}/{args.b_lr}_{args.b_batch}_{args.b_epoch}{args.subforder}'
        if args.focal_loss:
            file_path = file_path

        if not os.path.exists(file_path):
            os.mkdir(file_path)
        save = open(f'{file_path}/orbit_weight.pickle', 'wb')
        pickle.dump(orbit_weight_list, save)
        orbit_weight_info = open(f'{file_path}/orbit_weight.txt', 'w')
        if not(args.reg):
            for orbit, acc, simple_acc in zip(self.use_orbit, self.orbit_weight_info, self.new_orbit_acc):
                orbit_weight_info.write(f'orbit: {orbit}, acc: {acc}, simple_acc: {simple_acc}\n')
            orbit_weight_info.write(f'mean acc: {sum(self.orbit_weight_info)/len(self.orbit_weight_info)}')
            #orbit_weight_info.write(f'mean acc: {sum(self.orbit_weight_info) /1}')



def orbit_acc(y, use_orbit):
    n_orbit = len(use_orbit)
    n_zero = torch.sum((y==0).long(), dim=0)
    n_one = torch.sum((y == 1).long(), dim=0)

    orbit_accu = torch.max(n_zero, n_one)/y.shape[0]
    return orbit_accu

class linear(torch.nn.Module):
    def __init__(self, n_select_orbit):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn((n_select_orbit, 1), requires_grad=True))
    def forward(self,x):
        self.x = torch.exp(self.weight)

        return torch.matmul(x, self.x)

class score_learning:
    def __init__(self, orbit_weight, use_orbit, model_weight, label, answer, representation):
        #orbit_weight.shape = hidden x used_orbit
        self.orbit_weight = orbit_weight
        self.use_orbit = use_orbit
        #model_weight.shape = class_num x hidden
        self.model_weight = model_weight
        #input: orbit num, output: orbit_idx
        self.orbit_idx_dict = {v:k for k, v in enumerate(use_orbit)}
        self.result = []
        self.label = label
        self.answer = answer
        self.representation = representation

    def train(self, class_idx, target_orbit, representation,  last = False):
        #class_weight.shape = hidden
        class_weight = torch.unsqueeze(self.model_weight[class_idx, :], dim = -1)

        target_orbit_idx = [self.orbit_idx_dict[t] for t in target_orbit]
        #slect_orbit_weight.shape = hidden x n_target_orbit
        select_orbit_weight = self.orbit_weight[:, target_orbit_idx]

        #node_score.shape = num_answer_node x select_orbit
        node_score = torch.matmul(representation, select_orbit_weight)


        min_loss = 1000000000000000

        model = linear(len(target_orbit))
        answer_score = 0
        optimizer = torch.optim.Adam(params=model.parameters(), lr = args.s_lr)
        #loss_f = torch.nn.MSELoss(reduction='sum')
        loss_f = torch.nn.MSELoss()
        for epoch in range(1, 1+args.s_epoch):
            predict = model(select_orbit_weight)
            #model.x.shape = used_orbit x 1
            # print('predict', predict[:10])
            # print(class_weight[:10])
            loss = loss_f(predict, class_weight)
            #loss1 = loss_f(predict, class_weight)
            #loss2 = torch.mean(torch.nn.functional.relu(-1*torch.mul(node_score, torch.squeeze(model.x))))*100

            # if class_idx == 1 and epoch%200 == 0:
            #      print(loss1, loss2, target_orbit)
            #loss = loss1+loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if  min_loss > loss.item():
                answer_score = model.x.detach()
            min_loss = min(min_loss, loss.item())

        if last:
            predict = model(select_orbit_weight)
            #print('orbit :', select_orbit_weight[:10])
            print('hh:', model.x[:, 0])
            return answer_score, min_loss
        return min_loss

    def select(self, class_idx):
        #class_weight.shape = hidden x 1
        class_weight = self.model_weight[class_idx, :]


        reference_class_weight = torch.unsqueeze(class_weight/torch.sum(class_weight**2)**(1/2), dim=0)
        orbit_weight = self.orbit_weight
        similarty = torch.matmul(reference_class_weight, orbit_weight)
        args_similarity = torch.argsort(similarty, descending=True)[0,:].tolist()

        class_ansewr_node = torch.where(torch.mul((self.label == class_idx).long(), self.answer.long()) == 1)[0]
        #representation.shape = ansewr_node x hidden
        representation = self.representation[class_ansewr_node].detach()

        if args.agg:
            path = f'result/{args.data}/agg_model/{args.model_name}/{args.b_lr}_{args.b_batch}_{args.b_epoch}{args.subforder}/'
        else:
            path = f'result/{args.data}/model/{args.model_name}/{args.b_lr}_{args.b_batch}_{args.b_epoch}{args.subforder}/'
        if class_idx == 0:
            result_txt = open(f'{path}/similarity.txt', 'a')
        else:
            result_txt = open(f'{path}/similarity.txt', 'a')
        show_top = 10
        print(f'In {class_idx} class similarity\n')
        result_txt.write('\n')
        result_txt.write(f'class {class_idx}\n')
        for idx in args_similarity:
            print(f'orbit {self.use_orbit[idx]}:, {similarty[:,idx].item()}')
            result_txt.write(f'orbit {self.use_orbit[idx]}:, {similarty[:,idx].item()}\n')
            show_top -= 1
            if show_top == 0:
                break

        #choice_orbit = [self.use_orbit[args_similarity[0]]]

        #min_loss = self.train(class_idx, choice_orbit, representation)
        #choice_orbit = []
        choice_orbit = []
        min_loss = 10000
        #print(f'In {class_idx} class, select the {choice_orbit[0]} orbit, loss: {min_loss}')
        n_id = 0
        for _ in range(args.concept_n-1):
            selected_orbit = -1
            # if n_id <= 2:
            #     min_loss = 1000
            for o in self.use_orbit:

                if o in choice_orbit:
                    continue
                now_loss = self.train(class_idx, choice_orbit + [o], representation)
                # print(class_weight.shape)
                # print(class_weight.shape)
                if min_loss > now_loss:
                    min_loss = now_loss
                    selected_orbit = o
            #if selected_orbit == -1 and n_id >= 3:
            if selected_orbit == -1:
                print(f'In {class_idx} class, selecting orbit is end\n')
                break
            else:
                print(f'In {class_idx} class, select the {selected_orbit} orbit, loss: {min_loss}')
                n_id += 1
                choice_orbit.append(selected_orbit)
            print(f'In {class_idx} class, select the {selected_orbit} orbit, loss: {min_loss}')
            #choice_orbit.append(selected_orbit)

        choice_orbit.sort()
        print('-----------')
        selected_orbit_score, min_loss = self.train(class_idx, choice_orbit,representation, True)
        print(min_loss)
        target_orbit_idx = [self.orbit_idx_dict[t] for t in choice_orbit]
        #slect_orbit_weight.shape = hidden x n_target_orbit
        select_orbit_weight = self.orbit_weight[:, target_orbit_idx]
        #요기요기
        # print(min_loss)
        # print('orbit :',select_orbit_weight.shape)
        # print(class_weight.shape)
        # print(selected_orbit_score.shape)
        #
        # mes = torch.nn.MSELoss(reduction='sum')
        # print(torch.unsqueeze(class_weight, dim=-1).shape)
        # print(torch.sum((select_orbit_weight*selected_orbit_score-torch.unsqueeze(class_weight, dim=-1))**2))
        # print(torch.sum((select_orbit_weight * selected_orbit_score - class_weight) ** 2))
        # print(mes(select_orbit_weight*selected_orbit_score, class_weight))
        # exit()


        return selected_orbit_score, min_loss, choice_orbit

    def run(self):
        class_num = self.model_weight.shape[0]
        for c in range(class_num):
            selected_orbit_score, min_loss, choice_orbit = self.select(c)
            self.result.append([selected_orbit_score, min_loss, choice_orbit])
        if args.agg:
            path = f'result/{args.data}/agg_model/{args.model_name}/{args.b_lr}_{args.b_batch}_{args.b_epoch}{args.subforder}/{args.s_lr}_{args.s_epoch}'
        else:
            path = f'result/{args.data}/model/{args.model_name}/{args.b_lr}_{args.b_batch}_{args.b_epoch}{args.subforder}/{args.s_lr}_{args.s_epoch}'
        if args.focal_loss:
            path = f'result/{args.data}/model/{args.model_name}/{args.b_lr}_{args.b_batch}_{args.b_epoch}{args.subforder}/{args.s_lr}_{args.s_epoch}'
        if not(os.path.exists(path = f'{path}')):
            os.mkdir(f'{path}')
        result = open(f'{path}/result_sl.pickle', 'wb')
        pickle.dump(self.result, result)

class result_analyzer():
    def __init__(self, score_learning_result, orbit_weight, use_orbit,  same_dict, representation, label, ansewr, model_score_wo_bias, graph, y, model, feature, edge, orbit):
        self.y = y
        self.score_learning_result = score_learning_result

        self.orbit_weight = orbit_weight
        self.use_orbit = use_orbit
        self.representation = representation
        self.label = label
        self.answer = ansewr
        self.model_score_wo_bias = model_score_wo_bias
        self.graph = graph
        self.model  = model
        self.feature = feature
        self.edge = edge
        self.edge_matrix = torch.zeros((self.feature.shape[0], self.feature.shape[0]))
        self.same_dict = same_dict
        self.orbit_find_dict = [0,1,find_2_orbit,find_3_orbit,find_4_orbit,find_5_orbit,find_6_orbit,find_7_orbit,find_8_orbit,find_9_orbit,find_10_orbit,
                                find_11_orbit,find_12_orbit,find_13_orbit,find_14_orbit,find_15_orbit,find_16_orbit,find_17_orbit,find_18_orbit,find_19_orbit,find_20_orbit,find_21_orbit,find_22_orbit,find_23_orbit,
                                find_24_orbit,find_25_orbit,find_26_orbit,find_27_orbit,find_28_orbit,find_29_orbit,find_30_orbit,find_31_orbit,find_32_orbit,find_33_orbit,find_34_orbit,find_35_orbit,find_36_orbit,find_37_orbit,find_38_orbit,find_39_orbit,find_40_orbit,
                                find_41_orbit,find_42_orbit,find_43_orbit,find_44_orbit,find_45_orbit,find_46_orbit,find_47_orbit,
                                find_48_orbit,find_49_orbit,find_50_orbit,find_51_orbit,find_52_orbit,find_53_orbit,find_54_orbit,find_55_orbit,find_56_orbit,find_57_orbit,find_58_orbit,find_59_orbit,find_60_orbit,find_61_orbit,find_62_orbit,find_63_orbit,find_64_orbit,find_65_orbit,find_66_orbit,find_67_orbit,find_68_orbit,find_69_orbit,find_70_orbit,find_71_orbit,find_72_orbit]
        for i in range(edge.shape[1]):
            self.edge_matrix[edge[0, i], edge[1, i]] = 1
            self.edge_matrix[edge[1, i], edge[0, i]] = 1
        self.orbit_tensor = orbit

    def class_score(self, class_idx):

        if args.model_name.startswith('ppi') or args.model_name.startswith('model') :
            selected_orbit_score, min_loss, choice_orbit = self.score_learning_result[0]
        else:
            selected_orbit_score, min_loss, choice_orbit = self.score_learning_result[class_idx]
        class_score = selected_orbit_score/torch.sum(selected_orbit_score)
        print('----------------------------------------')
        print(f'In {class_idx} class, score order is\n')
        class_score_order = torch.argsort(class_score, descending=True, dim=0)
        if args.agg:
            result_path = f'result/{args.data}/agg_model'
        else:
            result_path = f'result/{args.data}/model'
        if class_idx == 0:
            result_txt = open(f'{result_path}/{args.model_name}/{args.b_lr}_{args.b_batch}_{args.b_epoch}{args.subforder}/{args.s_lr}_{args.s_epoch}/result_txt', 'a')
        else:
            result_txt = open(
                f'{result_path}/{args.model_name}/{args.b_lr}_{args.b_batch}_{args.b_epoch}{args.subforder}/{args.s_lr}_{args.s_epoch}/result_txt',
                'a')
        result_txt.write('\n')
        result_txt.write(f'class: {class_idx}\n')
        for i in class_score_order:
            print(f'{choice_orbit[i]}: {class_score[i, :].item()}')
            result_txt.write(f'{choice_orbit[i]}: {class_score[i, :].item()}\n')


    def make_used_orbit_weight(self, class_idx, orbit_weight, use_orbit):
        _, _, choice_orbit = self.score_learning_result[class_idx]

        choice_orbit_idx = [use_orbit.index(i) for i in choice_orbit]

        return orbit_weight[:, choice_orbit_idx]

    def node_class_score(self, class_idx):
        if args.model_name.startswith('ppi') or args.model_name.startswith('model') :
            used_orbit_weight = self.make_used_orbit_weight(0, self.orbit_weight, self.use_orbit)
            selected_orbit_score, _, choice_orbit = self.score_learning_result[0]
            model_score_wo_bias = self.model_score_wo_bias[:, 0]
        else:
            used_orbit_weight = self.make_used_orbit_weight(class_idx, self.orbit_weight, self.use_orbit)
            selected_orbit_score, _, choice_orbit = self.score_learning_result[class_idx]
            model_score_wo_bias = self.model_score_wo_bias[:, class_idx]

        node_orbit_score = torch.matmul(self.representation, used_orbit_weight)
        class_ansewr_node = torch.where(torch.mul((self.label == class_idx).long(), self.answer.long()) == 1)[0]


        #answer_node_orbit_score = node_orbit_score[class_ansewr_node]
        if args.agg:
            result_txt = open(
            f'result/{args.data}/agg_model/{args.model_name}/{args.b_lr}_{args.b_batch}_{args.b_epoch}{args.subforder}/{args.s_lr}_{args.s_epoch}/result_txt',
            'a')
        else:
            result_txt = open(
            f'result/{args.data}/model/{args.model_name}/{args.b_lr}_{args.b_batch}_{args.b_epoch}{args.subforder}/{args.s_lr}_{args.s_epoch}/result_txt',
            'a')
        num_node = len(class_ansewr_node)
        top1 = 0
        top3 = 0
        top5 = 0
        total_fider = 0
        total_sparsity = 0
        total_acc = 0
        total_length = 0
        ground_list = set()
        number_of_node = 0
        if args.ground_orbit:
            answer_orbit = args.ground_orbit[class_idx]
        #for n in range(num_node):
        for n in range(0, num_node):

            node = class_ansewr_node[n]
            if node%10 !=0:
                continue
            subgraph = k_hop_subgraph(node.item(), 3, self.edge)[1]
            if subgraph.shape[1] == 0:
                continue
            number_of_node += 1
            #if node in [4, 44, 128]:
            now_node_score = node_orbit_score[node,:]
            node_score_wo_bias = model_score_wo_bias[node]
            #now_node_orbit_score.shape = used_orbit
            now_node_orbit_score = torch.mul(now_node_score, torch.squeeze(selected_orbit_score)).detach()
            now_node_orbit_order = torch.argsort(now_node_orbit_score, descending=True)
            now_node_orbit_order = [choice_orbit[i] for i in now_node_orbit_order]
            predict_orbit = now_node_orbit_order

            subgraph = k_hop_subgraph(node.item(), 3, self.edge)[1]
            f, gp, sp, acc, lent = self.edge_detect(self.graph,node, self.label, self.use_orbit, self.y, predict_orbit)
            if sp == 0:
                number_of_node -= 1

            else:
                total_fider +=f
                #total_sparsity += (self.edge.shape[1]-sp)/self.edge.shape[1]
                total_sparsity += (subgraph.shape[1] - sp) / subgraph.shape[1]
                #total_sparsity += 1
            print(f)
            total_acc += acc
            total_length += lent
            ground_list.add(gp)
            #if predict_orbit != 10 and  predict_orbit != 28 and predict_orbit != 2 and predict_orbit != 57:

            if answer_orbit == now_node_orbit_order[0]:
                top1 += 1
            if answer_orbit in now_node_orbit_order[:3]:
                top3 += 1
            if answer_orbit in now_node_orbit_order[:5]:
                top5 += 1
        if number_of_node == 0:
            return 0, 0, 0, 0, 0, 0, 0

        print(f'In {class_idx} top1: {top1/num_node}, top3: {top3/num_node}, top5: {top5/num_node}')
        print(f'ground_orbit: {ground_list}')
        #result_txt.write(f'------top1: {top1/num_node}, top3: {top3/num_node}, top5: {top5/num_node}, sparsity: {total_sparsity/num_node}, acc: {total_acc/num_node}, length: {total_length/num_node}------\n')
        result_txt.write(
            f'------top1: {top1 / number_of_node}, top3: {top3 / number_of_node}, top5: {top5 / number_of_node}, sparsity: {total_sparsity / number_of_node}, acc: {total_acc / number_of_node}, length: {total_length / number_of_node}, fid: {total_fider / number_of_node}------\n')

        #return top1/num_node,top3/num_node, top5/num_node, total_fider/num_node, total_sparsity/num_node, total_acc/num_node, total_length/num_node
        return top1 / number_of_node, top3 / number_of_node, top5 / number_of_node, total_fider / number_of_node, total_sparsity / number_of_node, total_acc / number_of_node, total_length /number_of_node

    def run(self):
        class_num = torch.max(self.label).item()+1
        result_list = []
        total_top1 = 0
        total_top3 = 0
        total_top5 = 0
        total_fider = 0
        total_spar = 0
        total_length = 0
        total_acc = 0
        if args.agg:
            result_txt = open(
            f'result/{args.data}/agg_model/{args.model_name}/{args.b_lr}_{args.b_batch}_{args.b_epoch}{args.subforder}/{args.s_lr}_{args.s_epoch}/result_txt',
            'a')
        else:
            result_txt = open(
            f'result/{args.data}/model/{args.model_name}/{args.b_lr}_{args.b_batch}_{args.b_epoch}{args.subforder}/{args.s_lr}_{args.s_epoch}/result_txt',
            'a')
        not_using_class = 0

        for class_idx in range(0,class_num):
            #다른거 실험할 때 여기 퓔기
            print(class_idx)
            print('sdfdsfsdf')
            if class_idx ==0 or class_idx == 4:
                not_using_class += 1
                continue
            self.class_score(class_idx)
            top1, top3, top5, fider, spar, acc, length = self.node_class_score(class_idx)
            if fider == 0:
                not_using_class += 1
            total_fider += fider
            total_spar += spar
            total_top1 += top1
            total_top3 += top3
            total_top5 += top5
            total_acc += acc
            total_length += length
            result_list.append((top1, top3, top5, total_fider, total_length))
        class_num -= not_using_class
        print(class_num)
        print(f'------total result------\n top1: {total_top1/class_num}, top3: {total_top3/class_num}, top5: {total_top5/class_num}, fider: {total_fider/class_num}, spar: {total_spar/class_num}, acc: {total_acc/class_num}')
        result_txt.write(f'------total result------\n top1: {total_top1/class_num}, top3: {total_top3/class_num}, top5: {total_top5/class_num}, spar: {total_spar/class_num}, acc: {total_acc/class_num}, fider: {total_fider/class_num}, length: {total_length/class_num}')

    def edge_detect(self, graph, target_node, label, use_orbit, y, predicted_order):
        #print(fidelity(self.model, 60, self.feature, self.edge, [(182, 60),(347, 182), (207, 60), (78, 207), (944,60),(82, 60),(561, 60), (60, 182), (702, 60), (32, 60), (182, 347), (849, 60) ], label))
        if predicted_order == []:
            fiderity = 0
            predicted_edges = []
        acc = 0
        predicted_groud_truth = predicted_order[0]

        if args.data == 'bashapes' or args.data == 'bac':

            ground_edge = self.find_baco_gt(graph, target_node, label)

            if predicted_groud_truth in [57, 56, 58, 10, 28]:
                predicted_edges = self.find_baco_gt(graph, target_node, label)

                fiderity = fidelity(self.model, target_node, self.feature, self.edge, predicted_edges, label)
            else:
                fiderity, predicted_edges = self.orbit_detector(self.same_dict, graph, target_node, predicted_order,
                                                                label)
                #
            #
            #
            # elif predicted_groud_truth in [2,7, 11]:
            #     fiderity, predicted_edges = find_7_orbit(graph, target_node, label, self.model, self.feature,
            #                                                   self.edge, self.edge_matrix)
            #     if fiderity ==0:
            #         fiderity, predicted_edges = find_2_orbit(graph, target_node, label, self.model, self.feature,
            #                                                  self.edge)
            #
            # elif predicted_groud_truth in [3]:
            #     fiderity, predicted_edges = find_9_orbit(graph, target_node, label, self.model, self.feature, self.edge, self.edge_matrix)
            #     if fiderity == 0:
            #         fiderity, predicted_edges = find_3_orbit(graph, target_node, label, self.model, self.feature, self.edge)
            # elif predicted_groud_truth in [55]:
            #     fiderity, predicted_edges = find_55_orbit(graph, target_node, label, self.model, self.feature, self.edge, self.edge_matrix)
            # elif predicted_groud_truth in [8]:
            #     fiderity, predicted_edges = find_8_orbit(graph, target_node,label, self.model, self.feature, self.edge, self.edge_matrix)
            #
            # elif predicted_groud_truth in [16]:
            #     fiderity, predicted_edges = find_16_orbit(graph, target_node, label, self.model, self.feature, self.edge,
            #                                              self.edge_matrix)
            # elif predicted_groud_truth in [4]:
            #     fiderity, predicted_edges = find_17_orbit(graph, target_node, label, self.model, self.feature, self.edge,
            #                                              self.edge_matrix)
            #     if fiderity == 0:
            #         fiderity, predicted_edges = find_4_orbit(graph, target_node, label, self.model, self.feature,
            #                                                  self.edge, self.edge_matrix)
            # elif predicted_groud_truth in [26]:
            #     fiderity, predicted_edges = find_26_orbit(graph, target_node, label, self.model, self.feature,
            #                                               self.edge, self.edge_matrix)
            # elif predicted_groud_truth in [31]:
            #     fiderity, predicted_edges = find_31_orbit(graph, target_node, label, self.model, self.feature,
            #                                               self.edge, self.edge_matrix)
            # elif predicted_groud_truth in [32]:
            #     fiderity, predicted_edges = find_32_orbit(graph, target_node, label, self.model, self.feature,
            #                                               self.edge, self.edge_matrix)
            # elif predicted_groud_truth in [72]:
            #     fiderity, predicted_edges = find_72_orbit(graph, target_node, label, self.model, self.feature, self.edge,
            #                                              self.edge_matrix)
            # elif predicted_groud_truth in [22]:
            #     fiderity, predicted_edges = find_22_orbit(graph, target_node, label, self.model, self.feature, self.edge,
            #                                              self.edge_matrix)
            # elif predicted_groud_truth in [33]:
            #     fiderity, predicted_edges = find_33_orbit(graph, target_node, label, self.model, self.feature, self.edge,
            #                                              self.edge_matrix)
            # elif predicted_groud_truth in [43]:
            #     fiderity, predicted_edges = find_43_orbit(graph, target_node, label, self.model, self.feature, self.edge,
            #                                              self.edge_matrix)
            # elif predicted_groud_truth in [23]:
            #     fiderity, predicted_edges = find_23_orbit(graph, target_node, label, self.model, self.feature, self.edge,
            #                                              self.edge_matrix)
            # elif predicted_groud_truth in [11]:
            #     fiderity, predicted_edges = find_11_orbit(graph, target_node, label, self.model, self.feature, self.edge)
            # elif predicted_groud_truth in [69]:
            #     fiderity, predicted_edges = find_69_orbit(graph, target_node, label, self.model, self.feature, self.edge,
            #                                              self.edge_matrix)

            # else:
            #     print(predicted_groud_truth)
            #     print('fuck')
            #     exit()
            #     fiderity = 0
            #     sparsity = 0
            sparsity = len(predicted_edges)


        # elif args.data == 'bac':
        #     ground_edge = self.find_baco_gt(graph, target_node, label)
        #
        #
        #     print(predicted_groud_truth, target_node)
        #     if predicted_groud_truth in [57, 56, 58]:
        #         predicted_edges = self.find_baco_gt(graph, target_node, label)
        #         fiderity = self.fidelity(self.model, target_node, self.feature, self.edge, predicted_edges, label)
        #
        #     elif predicted_groud_truth in [2, 7]:
        #         fiderity, predicted_edges = find_7_orbit(graph, target_node, label, self.model, self.feature, self.edge, self.edge_matrix, 100)
        #         if fiderity == 0:
        #             fiderity, predicted_edges = find_2_orbit(graph, target_node, label, self.model, self.feature,
        #                                                       self.edge, self.edge_matrix)
        #     elif predicted_groud_truth in [3]:
        #         fiderity, predicted_edges = find_9_orbit(graph, target_node, label, self.model, self.feature, self.edge, self.edge_matrix, 100)
        #         if fiderity == 0:
        #             fiderity, predicted_edges = find_3_orbit(graph, target_node, label, self.model, self.feature, self.edge, 100)
        #
        #     elif predicted_groud_truth in [52]:
        #         fiderity, predicted_edges = find_52_orbit(graph, target_node, label, self.model, self.feature, self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [4, 17]:
        #         fiderity, predicted_edges = find_17_orbit(graph, target_node, label, self.model, self.feature,
        #                                                   self.edge,
        #                                                   self.edge_matrix, 100)
        #
        #         if fiderity == 0:
        #             fiderity, predicted_edges = find_4_orbit(graph, target_node, label, self.model, self.feature,
        #                                                      self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [5]:
        #         fiderity, predicted_edges = find_33_orbit(graph, target_node, label, self.model, self.feature,
        #                                                  self.edge, self.edge_matrix, 100)
        #         if fiderity == 0:
        #             fiderity, predicted_edges = find_5_orbit(graph, target_node, label, self.model, self.feature,
        #                                                      self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [8]:
        #         fiderity, predicted_edges = find_8_orbit(graph, target_node, label, self.model, self.feature, self.edge,
        #                                                  self.edge_matrix, 100)
        #     elif predicted_groud_truth in [11]:
        #         fiderity, predicted_edges = find_11_orbit(graph, target_node, label, self.model, self.feature,
        #                                                   self.edge, 100)
        #
        #     elif predicted_groud_truth in [22]:
        #         fiderity, predicted_edges = find_22_orbit(graph, target_node, label, self.model, self.feature,
        #                                                   self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [28]:
        #         fiderity, predicted_edges = find_28_orbit(graph, target_node, label, self.model, self.feature,
        #                                                   self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [10]:
        #         fiderity, predicted_edges = find_29_orbit(graph, target_node, label, self.model, self.feature,
        #                                                   self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [19]:
        #         fiderity, predicted_edges = find_19_orbit(graph, target_node, label, self.model, self.feature,
        #                                                   self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [21]:
        #         fiderity, predicted_edges = find_21_orbit(graph, target_node, label, self.model, self.feature,
        #                                                   self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [26]:
        #         fiderity, predicted_edges = find_26_orbit(graph, target_node, label, self.model, self.feature,
        #                                                   self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [27]:
        #         fiderity, predicted_edges = find_27_orbit(graph, target_node, label, self.model, self.feature,
        #                                                   self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [13]:
        #         fiderity, predicted_edges = find_13_orbit(graph, target_node, label, self.model, self.feature,
        #                                                   self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [65]:
        #         fiderity, predicted_edges = find_65_orbit(graph, target_node, label, self.model, self.feature,
        #                                                   self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [71]:
        #         fiderity, predicted_edges = find_71_orbit(graph, target_node, label, self.model, self.feature,
        #                                                   self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [12]:
        #         fiderity, predicted_edges = find_12_orbit(graph, target_node, label, self.model, self.feature,
        #                                                   self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [14]:
        #         fiderity, predicted_edges = find_14_orbit(graph, target_node, label, self.model, self.feature,
        #                                                   self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [13]:
        #         fiderity, predicted_edges = find_13_orbit(graph, target_node, label, self.model, self.feature,
        #                                                   self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [70]:
        #         fiderity, predicted_edges = find_70_orbit(graph, target_node, label, self.model, self.feature,
        #                                               self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [68]:
        #         fiderity, predicted_edges = find_68_orbit(graph, target_node, label, self.model, self.feature,
        #                                               self.edge, self.edge_matrix, 100)
        #     elif predicted_groud_truth in [23]:
        #         fiderity, predicted_edges = find_23_orbit(graph, target_node, label, self.model, self.feature,
        #                                               self.edge, self.edge_matrix, 100)
        #     else:
        #         print('fuck')
        #         print(predicted_groud_truth)
        #         exit()
        #         fiderity = 0.5
        #         sparsity = 10
        #         acc = 8
            if predicted_edges == []:
                print('run')
                return self.edge_detect( graph, target_node, label, use_orbit, y, predicted_order[1:])

            sparsity = len(predicted_edges)

            for j in predicted_edges:
                if j in ground_edge:
                    acc += 1
            acc = acc/len(ground_edge)




        else:
            fiderity, predicted_edges = self.orbit_detector(self.same_dict, graph, target_node, predicted_order, label)
            # print(predicted_edges)
            # print(fiderity)

            if predicted_edges == []:
                print('run')
                if len(predicted_order) == 1:
                    return 0, 0, 0, 0, 0
                return self.edge_detect( graph, target_node, label, use_orbit, y, predicted_order[1:])
            sparsity = len(predicted_edges)
            print(predicted_edges)
            acc = 0

        return fiderity, predicted_groud_truth, sparsity, acc, len(predicted_edges)

    def orbit_detector(self, same_dict, graph, target_node, predict_order, label):

        orbit = predict_order[0]
        print(f'node: {target_node}')
        answer_orbit = same_dict[orbit] if orbit in same_dict.keys() else [orbit]
        if orbit not in answer_orbit:
            answer_orbit.append(orbit)

        max_fiderity = -50000
        predicted_edges = []
        for target_orbit in answer_orbit:
            for i in range(args.sample):

            # if self.orbit_tensor[target_node, target_orbit] > 1000:
            #     print(target_orbit)
            #     print(self.orbit_tensor[target_node, target_orbit])
            #     continue
            #exec(f'fiderity, predicted_edge = find_{target_orbit}_orbit(graph, target_node, label,self.model, self.feature,self.edge, self.edge_matrix)')
                fiderity, predicted_edge = self.orbit_find_dict[target_orbit](graph, target_node, label,self.model, self.feature,self.edge, self.edge_matrix, args.n)

                if fiderity > max_fiderity:
                    max_fiderity = fiderity
                    predicted_edges = predicted_edge

        return max_fiderity, predicted_edges



    def find_baco_gt(self, graph, node, label):

        if label[node] == 0 or label[node] == 4:
            return
        if label[node] in [5, 7, 6]:
            ground_dict = {5: 0, 6: 0, 7: 1}
        else:
            ground_dict = {1: 0, 2: 0, 3: 1}
        neighbor_nodes = deque([node.item()])
        ground_nodes = []
        ground_edges = []
        while True:

            now_node = neighbor_nodes.popleft()

            now_label = label[now_node].item()

            if now_label in ground_dict.keys():
                if ground_dict[now_label] < 2:
                    ground_nodes.append(now_node)
                    ground_dict[now_label] += 1

                    neigbor = list(graph.neighbors(now_node))

                    neighbor_nodes += neigbor

                    for j in neigbor:
                        if label[j].item() != 0 and label[j].item() != 4 and abs(j-now_node)<10 :
                            ground_edges.append((now_node, j))
                if sum(ground_dict.values()) == 6:
                    if len(ground_edges) == 12:
                        return ground_edges







    # def fidelity(self, model, target_node, feature, edge, explaine_edge, label):
    #     with torch.no_grad():
    #         answer_predict = model(feature, edge)
    #         answer_predict = torch.softmax(answer_predict, dim=1)[target_node, label[target_node]]
    #         tuple_edge = []
    #         edge_index = [i for i in range(edge.shape[1])]
    #         edge_weights = torch.ones(edge.size(1))
    #         for i in range(edge.shape[1]):
    #             a, b = edge[:, i]
    #             tuple_edge.append((a.item(), b.item()))
    #
    #         for i in explaine_edge:
    #             i_edge = tuple_edge.index(i)
    #             edge_index.remove(i_edge)
    #             edge_weights[i_edge] =0
    #         new_edge = edge[:, edge_index]
    #         processing_predict = model(feature, edge, edge_weights = edge_weights)
    #         processing_predict = torch.softmax(processing_predict, dim=1)[target_node, label[target_node]].item()
    #
    #         return answer_predict - processing_predict




class FocalLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=3,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = torch.nn.functional.binary_cross_entropy(input, target,reduction=self.reduction)

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss



class ohemloss_f(torch.nn.Module):
    def __init__(self):
        super(ohemloss_f, self).__init__()
        self.loss_f = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predict, y):
        loss = self.loss_f(predict, y)
        pred = (torch.sigmoid(predict) >= 0.5).int()
        zero_loss = torch.sort(loss[torch.where(pred == 0)[0], :], dim=0, descending=True)[0]
        one_loss = torch.sort(loss[torch.where(pred == 1)[0], :], dim=0, descending=True)[0]
        min_number = min(zero_loss.shape[0], one_loss.shape[0])
        final_loss = torch.tensor(0.)
        axis = 0
        for i in (zero_loss, one_loss):
            if i.shape[0] == min_number and i.shape[0] != 0:
                final_loss += torch.mean(i)
                axis += 1

            elif i.shape[0] != min_number:
                max_number = max(min_number, 3*(min_number+1))
                max_number = min(max_number, i.shape[0])
                final_loss += torch.mean(i[:max_number, 0])
                axis += 1

        return final_loss/axis

def processing_orbit(data, node_num, target_orbit ):
    path = f'data/{data}/'
    save_path = f'data/{data}/'
    orbit = pd.read_csv(f'{path}orbit.txt', sep = '\t', header=None)
    orbit.index = orbit[0]
    orbit = orbit.drop([0], axis=1)
    orbit.columns = [i for i in range(73)]
    zero = (orbit.loc[:,:] == 0).sum(axis = 0)
    nonzero = (orbit.loc[:, :] != 0).sum(axis=0)
    scaler = MinMaxScaler()
    scaler.fit(orbit)
    scaled_orbit = scaler.transform(orbit)
    scaled_orbit = pd.DataFrame(data=scaled_orbit, index=orbit.index, columns=orbit.columns)

    for i in range(73):
        print(f'orbit: {i}, zero: {zero[i]}, non_zero: {nonzero[i]}')

    feature = open(f'{save_path}features.txt', 'w')

    zero = 0
    one = 0
    two = 0
    for i in range(node_num):
        a1 = orbit.loc[i, target_orbit]

        a2 = orbit.loc[i, 59]
        if a1>0:
            label = 1
            one += 1
        elif a1 ==0:
            label =0
            zero+=1

        line = f"{i} 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 {label}\n"
            #line = f"{i} {scaled_orbit.loc[i,49]} {label}\n"

        feature.write(line)
    print(zero, one)


def make_graph(data):
    graph = nx.Graph()

    edge_list = open(f'data/{data}/edge_list.txt', 'r')
    for i in edge_list:
        node1, node2 = map(int, i.strip().split(' '))
        graph.add_edge(node1, node2)
    return graph


def extract_use_orbit(num_node, answer_nodes, use_orbit):
    data_path = f'data/{args.data}/orbit.txt'
    raw_orbit = np.genfromtxt(f'{data_path}', dtype=np.int32)
    orbit = np.zeros((num_node, 73))

    extracted_orbit = orbit[:, use_orbit]
    return extracted_orbit

#select_the_learnable_orbit()