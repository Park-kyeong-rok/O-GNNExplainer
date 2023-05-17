import os

score_search = False
b_epoch = [4000]
b_lr = [0]
b_batch = [1, 1, 1]
s_lr = [0.0015]
s_epoch = [3000]
for b_epochs in b_epoch:
    for b_lrs in b_lr:
        for b_batchs in b_batch:
            b_epochs = 5000
            # if b_batchs == 0:
            #     b_epochs = 9000
            # elif b_batchs == 1:
            #     b_epochs = 500
            # elif b_batchs == 4:
            #     b_epochs = 1000
            if b_batchs == 0:
                thr = 0.6060291528701782
            elif b_batchs == 1:
                thr = 0.3414493799209595
            elif b_batchs == 2:
                thr =0.25484681129455566
            elif b_batchs == 3:
                thr = 0.4225941002368927
            elif b_batchs == 4:
                thr = 0.28328025341033936
            elif b_batchs == 5:
                thr = 0.3360317647457123
            if not(score_search):

                #os.system(f'python3 run.py  --data ppi{b_batchs} --model_name ppi{b_batchs}_model --thr {thr}')
                os.system('python run.py' )

                continue
            else:
                for s_lrs in s_lr:
                    for s_epochs in s_epoch:
                        os.system(f'python run.py --b_epoch {b_epochs} --b_lr {b_lrs} --b_batch {b_batchs} --s_lr {s_lrs} --s_epoch {s_epochs}')




# s_lrs = [0.01, 0.001, 0.005]
# s_epochs = [500, 1000, 2000, 3000]
# for s_lr in s_lrs:
#     for s_epoch in s_epochs:
#         os.system(f'python run.py --s_lr {s_lr} --s_epoch {s_epoch}')