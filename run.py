
from arguments import args
import pickle
from utils import *
import numpy as np
import torch

model = load_model()
model.eval()
feature, label, edge = load_data(args.data)

#model_bias.shape = num_label, model_weight.shape = num_label x hidden_state
model_bias, model_weight = model.lin.bias.detach(), model.lin.weight.detach()

#model_score = num_node x num_class

model_score = model(feature, edge).detach()
# print(model_score)
# exit()
model_score_wo_bias = model_score - model_bias


if args.model_name.startswith('ppi') or args.model_name.startswith('model')  :

    predict = torch.squeeze((torch.sigmoid(model_score)>args.thr).long())

    answer = predict == label

    ansewr_node_idx = torch.where(answer == True)[0]
    representation = model.representation[answer].detach()
    n_node = model_score.shape[0]
    n_class = torch.unique(label).shape[0]

else:


    predict = torch.argmax(model_score, dim =1)

    answer = predict == label

    #삭제 요망
    #answer = torch.LongTensor([0, 1, 2, 3, 4])

    ansewr_node_idx = torch.where(answer==True)[0]

    #삭제 요망
    #answer_node_idx = torch.LongTensor([0, 1, 2, 3, 4])
    # print(torch.mean(answer.float()))
    # exit()
    #representation.shape = answer_num x hidden_state
    # print(model_bias)
    # print(model_score)
    # print(model(feature, edge))
    # exit()
    representation = model.representation[answer].detach()

    # label = label[answer]
    # label = torch.squeeze(torch.nonzero((label == 1).int()))

    #model_bias = model.
    n_node = model_score.shape[0]
    n_class = model_score.shape[1]

    # y = answer_node x use_orbit
data_path = f'data/{args.data}/orbit.txt'
raw_orbit = np.genfromtxt(f'{data_path}', dtype=np.int32)
orbit = np.zeros((n_node, 73))

for i in raw_orbit:
    orbit[i[0],:] = i[1:]



y, use_orbit, same_dict = load_orbit(num_node=n_node, reg = args.reg, answer = answer)

y = torch.FloatTensor(y)
orbit_acc = orbit_acc(y, use_orbit)

if args.agg:
    if not (os.path.exists(f'result/{args.data}/agg_model/{args.model_name}')):
        os.mkdir(f'result/{args.data}/agg_model/{args.model_name}/')
else:
    if not(os.path.exists(f'result/{args.data}/model/{args.model_name}')):
        os.mkdir(f'result/{args.data}/model/{args.model_name}/')



#orbit_weight 학습
if args.Orbit_weight_learning:
    binary_classifier = binary_classification(use_orbit, representation, y)
    binary_classifier.run(orbit_acc)

if args.agg:
    r_path = f'result/{args.data}/agg_model'
else:
    r_path = f'result/{args.data}/model'
result_path = f'{r_path}/{args.model_name}/{args.b_lr}_{args.b_batch}_{args.b_epoch}{args.subforder}'
if args.focal_loss:
    result_path = f'{r_path}/{args.model_name}/{args.b_lr}_{args.b_batch}_{args.b_epoch}'
orbit_weight_list = open(f'{result_path}/orbit_weight.pickle', 'rb')
use_orbit, orbit_weight, orbit_weight_regular = pickle.load(orbit_weight_list)
print(f'after orbit wegiht learning, we use orbit(total: {len(use_orbit)})------------')
print(f'{use_orbit}')

y = extract_use_orbit(num_node=n_node, answer_nodes = answer,use_orbit = use_orbit)
if not(args.Normalize):

    orbit_weight = orbit_weight_regular
# print(orbit_weight[:,-1])
# print(orbit_weight[:,-2])
# print(model_weight[-1,:])
if args.score_learning:
    score_learner = score_learning(orbit_weight, use_orbit, model_weight, label, answer, model.representation)

    score_learner.run()
graph = make_graph(args.data)

score_learning_result = open(f'{result_path}/{args.s_lr}_{args.s_epoch}/result_sl.pickle', 'rb')
score_learning_result = pickle.load(score_learning_result)

analyzer =  result_analyzer(score_learning_result,orbit_weight, use_orbit, same_dict, model.representation, label, answer ,model_score_wo_bias, graph, y, model, feature, edge, orbit)
analyzer.run()