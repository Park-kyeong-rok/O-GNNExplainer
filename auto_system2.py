import os
folder_list = os.listdir('data/')

for i in range(0, 73):
    data_name = f'ppi_random_{i}'
    if data_name in folder_list:
        txt = open(f'data/{data_name}/model.txt', 'r')
        thr, _ = txt.readlines()
        thr = float(thr.strip())
        os.system(
            f'python run.py  --data {data_name} --model_name model3 --thr {thr}')