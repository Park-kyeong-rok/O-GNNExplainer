import os
import copy
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch_geometric.data
from sklearn.metrics import roc_auc_score
from model import *
from utils import *
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

def run(data_name):
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    lr = 0.001
    epochs = 600
    hidden_size = 30
    orbit_num = 22
    layer = 3
    agg= False
    weight_decay = 0
    #여기 고치기


    #data_name = 'ppi_random_57'
    data_path = f'../{data_name}/'

    feature, label, edge = load_data(data_name)
    data = torch_geometric.data.Data(x=feature, edge_index=edge, y=label)

    node_num = data.x.shape[0]
    input_dim = data.x.shape[1]
    label = data.y.numpy()
    label_num = len(np.unique(label))
    print(label)
    train_mask, test_mask, train_label, test_label = train_test_split([i for i in range(node_num)], label, test_size=0.1, stratify=label, random_state=30)
    train_mask, validation_mask, _,_ = train_test_split(train_mask, train_label,test_size=1/9, stratify=train_label, random_state=200)
    #train_mask.to(device)
    #test_mask.to(device)
    #validation_mask.to(device)
    print(f'train: {len(train_mask)}, val: {len(validation_mask)}, test: {len(test_mask)}')

    if agg:
        model = eval(f'agg_GCN{layer}(input_dim, label_num, hidden_size)')
    else:
        model = eval(f'GCN{layer}(input_dim, label_num, hidden_size)')
        #model = eval(f'GCN{layer}(input_dim, 1, hidden_size)')
        #model = Netsimple(10, 10)
        #model = agg_GCN3(10, 1, 10)
    model.train()
    model.to(device)
    loss_f = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr =lr, weight_decay=weight_decay)
    loss_list = []
    val_loss_list = []
    val_acc = 0
    total_acc = 0
    val_loss = 100000
    last_train_acc = 0
    last_thr = 0
    for epoch in range(1,epochs+1):
        model.train()
        model.to(device)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.to(device)
        result = model(x,edge_index=  edge_index)
        # print('------------------')
        # print(result)

        loss = loss_f(torch.squeeze(result[train_mask]), y[train_mask])

        loss_list.append(loss.detach().to('cpu').numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            final = model(x, edge_index=edge_index)
            score = torch.argmax(torch.nn.functional.log_softmax(final, dim=1), dim=1) == y

            model.eval()
            final = model(x, edge_index=edge_index)

            score = torch.argmax(torch.nn.functional.log_softmax(final, dim=1), dim=1) == y

            model.eval()

            acc = torch.argmax(torch.nn.functional.log_softmax(result, dim= 1), dim=1)


            answer = acc == y
            train_acc = torch.mean(answer[train_mask].float())

            train_loss = loss_f(result[train_mask], y[train_mask])
            validation_loss = loss_f(result[validation_mask], y[validation_mask])
            val_loss_list.append(validation_loss.detach().to('cpu').numpy())
            validation_acc = torch.mean(answer[validation_mask].float())
            totalment_acc = torch.mean(answer.float())
            print(f'epoch: {epoch}, t_loss: {loss.item()}, t_acc{train_acc:.3f}, v_loss: {validation_loss.item()}, v_acc = {validation_acc:.3f}, total_acc: {totalment_acc}')
            #print(f'epoch {epoch},acc {train_acc}, loss {loss.item()}')
    #torch.save(model.state_dict(), f'../data/{data_name}/model/model')
           #if epoch >= 10 and val_loss >= validation_loss:
            if epoch >= 10 and totalment_acc>= last_train_acc:
                model.eval()
                final = model(x, edge_index=edge_index)

                score = torch.argmax(torch.nn.functional.log_softmax(final, dim=1), dim=1) == y
                print(torch.mean(score.float()))
                print('-------------------')
                val_acc = torch.mean(score[validation_mask].float()).item()
                train_acc = torch.mean(score[train_mask].float()).item()
                test_acc = torch.mean(score[test_mask].float()).item()
                total_list = len(test_mask) + len(train_mask) + len(validation_mask)
                totol_acc = torch.mean(score.float()).item()

                print(f'val_acc : {val_acc}')
                print(f'test_acc: {test_acc}')
                print(f'train_acc: {train_acc}')
                print(f'final acc: {total_acc}')

                #model.eval()

                last_train_acc = totalment_acc
                val_loss = validation_loss
                if agg:
                    model.to('cpu')
                    torch.save(model.state_dict(), f'../data/{data_name}/agg_model/{lr}_{epochs}_{layer}')
                else:
                    if not('model'in os.listdir(f'../data/{data_name}')):
                        os.mkdir(f'../data/{data_name}/model')
                    model.to('cpu')
                    torch.save(model.state_dict(), f'../data/{data_name}/model/{lr}_{epochs}_{layer}')


        # score = torch.argmax(torch.nn.functional.log_softmax(final, dim=1), dim=1) == data.y
        # print(torch.mean(score.float()))
        print('-------------------')

    plt.plot(loss_list)
    plt.plot(val_loss_list)
    plt.legend(['train', 'validation'])
    plt.show()
    if agg:
        model.load_state_dict(torch.load(f'../data/{data_name}/agg_model/{lr}_{epochs}_{layer}'))
    else:
        model.load_state_dict(torch.load(f'../data/{data_name}/model/{lr}_{epochs}_{layer}'))

    model.eval()
    model.to('cpu')
    print(data.x.device)
    print(data.edge_index.device)
    final = model(data.x, edge_index=data.edge_index)
    score = torch.argmax(torch.nn.functional.log_softmax(final, dim=1 ),dim=1) == data.y
    print(torch.mean(score.float()))
    print('-------------------')
    val_acc = torch.mean(score[validation_mask].float()).item()
    train_acc = torch.mean(score[train_mask].float()).item()
    test_acc = torch.mean(score[test_mask].float()).item()
    total_list = len(test_mask) + len(train_mask) + len(validation_mask)
    totol_acc = torch.mean(score.float()).item()

    print(f'val_acc : {val_acc}')
    print(f'test_acc: {test_acc}')
    print(f'train_acc: {train_acc}')
    print(f'final acc: {total_acc}')



run('last_fm')