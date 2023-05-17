import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch_geometric.data
from sklearn.metrics import roc_auc_score
from model import *
from utils import *
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

def run(data_name):
    lr = 0.001
    epochs = 1000
    hidden_size = 10
    orbit_num = 22
    layer = 2
    agg= False
    weight_decay = 0
    #여기 고치기


    data_name = 'ppi_mini'
    data_path = f'../{data_name}/'

    feature, label, edge = load_data(data_name)
    data = torch_geometric.data.Data(x=feature, edge_index=edge, y=label)

    node_num = data.x.shape[0]
    input_dim = data.x.shape[1]
    label = data.y.numpy()
    label_num = len(np.unique(label))

    train_mask, test_mask, train_label, test_label = train_test_split([i for i in range(node_num)], label, test_size=0.1, stratify=label, random_state=30)
    train_mask, validation_mask, _,_ = train_test_split(train_mask, train_label,test_size=1/9, stratify=train_label, random_state=200)

    print(f'train: {len(train_mask)}, val: {len(validation_mask)}, test: {len(test_mask)}')

    if agg:
        model = eval(f'agg_GCN{layer}(input_dim, label_num, hidden_size)')
    else:
        #model = eval(f'GCN{layer}(input_dim, label_num, hidden_size)')
        model = eval(f'GCN{layer}(input_dim, 1, hidden_size)')
        #model = Netsimple(10, 10)
        #model = agg_GCN3(10, 1, 10)

    #loss_f = torch.nn.CrossEntropyLoss()
    loss_f = torch.nn.BCEWithLogitsLoss()
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
        result = model(data.x,edge_index=  data.edge_index)

        #result = model()
        loss = loss_f(torch.squeeze(result[train_mask]), data.y[train_mask].float())
        #loss = loss_f(result[train_mask], data.y[train_mask])
        loss_list.append(loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = torch.squeeze(torch.sigmoid(torch.squeeze(result)))
            train_acc, thr = result_anal(data.y, pred)
            #fpr, tpr, thresholds = roc_curve(data.y, pred)
            # get the best threshold
            #J = tpr - fpr

            #ix = np.argmax(J)
            #best_thresh = thresholds[ix]

            #acc = torch.nn.functional.log_softmax(result, dim=1 )
            #acc = torch.mean(((torch.squeeze(result)>best_thresh).long() ==  data.y).float())

        #print(f' acc:{acc}, {data.y}')
            #answer = (pred>best_thresh).long()

            #answer = answer == data.y
            #train_acc = torch.mean(answer.float())

            #model.eval()

            #validation_loss = loss_f(result[validation_mask], data.y[validation_mask])
            #val_loss_list.append(validation_loss.detach().numpy())
            #validation_acc = torch.mean(answer[validation_mask].float())
            #totalment_acc = torch.mean(answer.float())

            #print(f'epoch: {epoch}, t_loss: {loss.item()}, t_acc{train_acc:.3f}, v_loss: {validation_loss.item()}, v_acc = {validation_acc:.3f}, total_acc: {totalment_acc}')
            print(f'epoch {epoch},acc {train_acc}, loss {loss.item()}')
    #torch.save(model.state_dict(), f'../data/{data_name}/model/model')
            if epoch >= 10 and train_acc >= last_train_acc:
                last_thr = thr
                #model.eval()
                last_train_acc = train_acc
                if agg:
                    torch.save(model.state_dict(), f'../data/{data_name}/agg_model/{lr}_{epochs}_{layer}')
                else:
                    if not('model'in os.listdir(f'../data/{data_name}')):
                        os.mkdir(f'../data/{data_name}/model')
                    torch.save(model.state_dict(), f'../data/{data_name}/model/model2')
            #if epoch%50 == 0:
            #    print(f'test_acc: {torch.mean(answer[test_mask].float()).item()}')

    plt.plot(loss_list)
    plt.plot(val_loss_list)
    plt.legend(['train', 'validation'])
    plt.show()
    if agg:
        model.load_state_dict(torch.load(f'../data/{data_name}/agg_model/{lr}_{epochs}_{layer}'))
    else:
        model.load_state_dict(torch.load(f'../data/{data_name}/model/model2'))
    model.train()
    final = model(data.x, edge_index=data.edge_index)
    print(last_thr)
    score = (torch.sigmoid(torch.squeeze(final))>last_thr) == data.y
    print('-------------------')
    val_acc = torch.mean(score[validation_mask].float()).item()
    train_acc = torch.mean(score[train_mask].float()).item()
    test_acc = torch.mean(score[test_mask].float()).item()
    total_list = len(test_mask) + len(train_mask) + len(validation_mask)
    totol_acc = torch.mean(score.float()).item()

    print(f'val_acc : {torch.mean(score[validation_mask].float()).item()}')
    print(f'test_acc: {torch.mean(score[test_mask].float()).item()}')
    print(f'train_acc: {torch.mean(score[train_mask].float()).item()}')
    print(f'final acc: {torch.mean((score.float())).item()}')
    txt = open(f'../data/{data_name}/model2.txt', 'w')
    txt.write(str(last_thr)+'\n')
    txt.writelines(str(torch.mean((score.float())).item()) + '\n')


forder = os.listdir('../data')
for i in range(0, 72):
    data_name = f'ppi_random_{i}'
    if data_name in forder:
        run(data_name)