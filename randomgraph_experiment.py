from utils import *
import os
def processing_orbit(data, node_num, target_orbit ):
    path = f'data/{data}/'
    #if not(f'randomgraph_{target_orbit}') in os.listdir('data'):
    #    os.mkdir(f'data/randomgraph_{target_orbit}')

    save_path = f'data/{data}_{target_orbit}/'
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
        if min(zero[i], nonzero[i]) >=5:
            name = f'data/ppi_random_{i}'
            #os.mkdir(name)
            #feature = open(f'{name}/features.txt', 'w')

            zero_ = 0
            one_ = 0
            two = 0
            for j in range(node_num):
                a1 = orbit.loc[j, i]

                if a1>0:
                    label = 1

                elif a1 ==0:
                    label =0
                line = f"{j} 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 {label}\n"
                 # line = f"{i} {scaled_orbit.loc[i,49]} {label}\n"

                #feature.write(line)

processing_orbit('randomgraph', 340, 2)




