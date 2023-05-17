import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str ,default='ppi0',
                    help='Interested Data')
parser.add_argument('--thr', type=float, default=0.3454708,
                   help='Write the threshehold')
parser.add_argument('--reg', action='store_true' ,default=False,
                    help='regression mode or not?')
parser.add_argument('--n', type=int ,default=1,
                    help='regression mode or not?')
parser.add_argument('--sample', type=int ,default=50,
                    help='regression mode or not?')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='Using CUDA training.')
parser.add_argument('--cuda_number', type=str, default=0,
                    help='CUDA allocation number.')
parser.add_argument('--model_name', type=str, default='ppi0_model',
                   help='Write the model name used')
parser.add_argument('--ground_orbit', type=int, default=[2,58, 57, 56, 2, 58,57,56, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
                   help='Write the model name used')
parser.add_argument('--agg', type=int, default=False,
                   help='Write the model name used')
parser.add_argument('--cuda_n', type=int, default=1,
                    help='Using CUDA training.')

parser.add_argument('--Orbit_weight_learning', action='store_true', default=False,
                    help='Do you want to train the orbit weight?')
parser.add_argument('--OHEM_loss', action='store_true', default=False,
                    help='Do you want to train the orbit weight?')
parser.add_argument('--b_lr', type=float, default=0.003,
                    help='Binary classifier learning rate')
parser.add_argument('--b_batch', type=int, default=2048,
                    help='Bainary classifier batch')
parser.add_argument('--b_epoch', type=int, default=1000,
                    help='Binary classifier epoch')

parser.add_argument('--focal_loss', action='store_true', default=False,
                    help='Do you want to train the orbit weight?')


parser.add_argument('--score_learning', action='store_true', default=False,
                    help='Do you want to train the score?')
parser.add_argument('--s_lr', type=float, default=0.005,
                    help='Score learner learning rate')
parser.add_argument('--s_epoch', type=int, default = 1000,
                    help='Score learner epoch')
parser.add_argument('--num_model_layer', type=int ,default=3,
                    help='Number of layer of GNN model')
parser.add_argument('--Normalize', action='store_true', default=True,
                    help = '1. True - normal'
                           '2. False - regular')
parser.add_argument('--concept_n', type=int, default=10,
                   help='Write the model name used')
parser.add_argument('--folder_name', type=str, default='b',
                    help='write the file name to save result')




#[10,57,56]
parser.add_argument('--percent', type=float, default=0,
                   help='Write the model name used')
parser.add_argument('--bias', action='store_true', default=False,
                    help='Do you want to train the orbit weight?')
parser.add_argument('--subforder', type=str, default='')

args = parser.parse_args()