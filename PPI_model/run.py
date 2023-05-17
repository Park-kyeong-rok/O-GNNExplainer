from utils import *
import torch
from model import *
from torch_geometric.data import DataLoader
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_curve
from torch_geometric.datasets import PPI
from numpy import argmax
import numpy as np
from torch_geometric.utils import remove_isolated_nodes


task_id = 5
device = torch.device("cuda:2" if torch.cuda.is_available() else torch.device("cpu"))

epochs = 300
learning_rate = 0.003
if task_id != 100:
    train_ppi = PPI('node_dataset/ppi/', transform=get_task_rm_iso(task_id))
    train_loader = DataLoader(train_ppi, 1, shuffle= True)
    val_ppi = PPI('node_dataset/ppi_val/',split='val', transform=get_task_rm_iso(task_id))
    val_loader = DataLoader(val_ppi, 1, shuffle= True)
    test_ppi = PPI('node_dataset/ppi_test/',split='test', transform=get_task_rm_iso(task_id))
    test_loader = DataLoader(test_ppi, 1, shuffle= True)

model = GCN3(50, 1, 200).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
loss_f = torch.nn.BCELoss()
max_acc = 0
selected_thr = 0
selected_epoch = 0
for epoch in range(epochs):
    model.train()
    train_loss = 0
    total_predict = []
    total_y = []
    for data in train_loader:
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

        y = data.y.to(device)
        total_y.append(y.detach().to('cpu'))
        predict = torch.sigmoid(torch.squeeze(model(x, edge_index), dim= -1))
        total_predict.append(predict.detach().to('cpu'))
        loss = loss_f(predict, y)
        train_loss += loss


    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    total_predict = torch.concat(total_predict)
    total_y = torch.concat(total_y)
    fpr, tpr, thr = roc_curve(total_y, total_predict)
    j = tpr-fpr
    ix = np.argmax(j)
    best_thr = thr[ix]

    train_acc = torch.mean(((total_predict > best_thr).float()==total_y).float())

    model.eval()
    val_total_predict = []
    val_total_y = []
    val_loss = 0
    test_total_predict = []
    test_total_y = []
    test_loss = 0

    with torch.no_grad():
        for data in val_loader:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)

            y = data.y.to(device)
            val_total_y.append(y.detach().to('cpu'))
            predict = torch.sigmoid(torch.squeeze(model(x, edge_index), dim=-1))
            val_total_predict.append(predict.detach().to('cpu'))
            loss = loss_f(predict, y)
            val_loss += loss

        val_total_predict = torch.concat(val_total_predict)
        val_total_y = torch.concat(val_total_y)
        val_acc = torch.mean(((val_total_predict > best_thr).float()==val_total_y).float())


        for data in test_loader:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)

            y = data.y.to(device)
            test_total_y.append(y.detach().to('cpu'))
            predict = torch.sigmoid(torch.squeeze(model(x, edge_index), dim=-1))
            test_total_predict.append(predict.detach().to('cpu'))
            loss = loss_f(predict, y)
            test_loss += loss

        test_total_predict = torch.concat(test_total_predict)
        test_total_y = torch.concat(test_total_y)
        test_acc = torch.mean(((test_total_predict > best_thr).float()==test_total_y).float())
        print(f'-----------------------------{epoch}-----------------------------')
        print(f'epoch: {epoch}, train_loss: {train_loss}, train_acc: {train_acc}, val_loss: {val_loss}, val_acc: {val_acc}, test_acc:{test_acc}')
        total_predict = torch.concat([total_predict, val_total_predict, test_total_predict])
        total_y = torch.concat([total_y, val_total_y, test_total_y])
        total_acc = torch.mean(((total_predict > best_thr).float()== total_y).float())
        print(f'total_acc: {total_acc}')
        if total_acc> max_acc:
            torch.save(model.state_dict(), f'./ppi{task_id}_model')
            max_acc = total_acc
            selected_thr = best_thr
            max_val = val_acc
            max_test = test_acc
            max_train = train_acc
            selected_epoch = epoch
            if total_acc > 0.99:
                break
model_parameter = torch.load(f'./ppi{task_id}_model')
model.load_state_dict(model_parameter)
model.eval()
train_ppi = PPI('node_dataset/ppi/', transform=get_task_rm_iso(task_id))
train_loader = DataLoader(train_ppi, 1)
val_ppi = PPI('node_dataset/ppi_val/',split='val', transform=get_task_rm_iso(task_id))
val_loader = DataLoader(val_ppi, 1)
test_ppi = PPI('node_dataset/ppi_test/',split='test', transform=get_task_rm_iso(task_id))
test_loader = DataLoader(test_ppi, 1)
graph_number = 0

print('finish')
print(f'total_acc: {max_acc}, train_acc: {max_train}, val_acc: {max_val}, test_acc: {max_test}, epoch: {selected_epoch}')
print(f'best_the: {selected_thr}\n\n\n\n')
for data in train_loader:
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    predict = torch.sigmoid(torch.squeeze(model(x, edge_index), dim=-1))
    acc = torch.mean(((predict > selected_thr).float()== y).float())
    print(f'the {graph_number}th acc: {acc}')
    print(f'edge_n: {edge_index.shape[1]}, node_n" {x.shape[0]}')
    print('-------------------------------------')
    graph_number += 1

for data in val_loader:
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    predict = torch.sigmoid(torch.squeeze(model(x, edge_index), dim=-1))
    acc = torch.mean(((predict > selected_thr).float()== y).float())
    print(f'the {graph_number}th acc: {acc}')
    print(f'edge_n: {edge_index.shape[1]}, node_n" {x.shape[0]}')
    print('-------------------------------------')
    graph_number += 1

for data in test_loader:
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    predict = torch.sigmoid(torch.squeeze(model(x, edge_index), dim=-1))
    acc = torch.mean(((predict > selected_thr).float()== y).float())
    print(f'the {graph_number}th acc: {acc}')
    print(f'edge_n: {edge_index.shape[1]}, node_n" {x.shape[0]}')
    print('-------------------------------------')
    graph_number += 1




