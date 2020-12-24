#%%
import pickle
import numpy as np
import os

# %%
alg_name = ['PLN','PLF','Prog_NN', 'DF_CNN','LwF','EWC','O-EWC','SI', 'Replay \n (increasing amount)', 'Replay \n (fixed amount)', 'None']
model_file = ['dnn0','uf10','Prog_NN','DF_CNN', 'LwF', 'EWC', 'OEWC', 'SI', 'offline', 'exact', 'None']
total_alg = 11
slots = 10
shifts = 6
time_info = [[] for i in range(total_alg)]
mem_info = [[] for i in range(total_alg)]

for alg in range(total_alg): 
    count = 0.0
    tmp = np.zeros(10,dtype=float)
    for slot in range(slots):
        for shift in range(shifts):
            if alg < 2:
                filename = './result/time_res/'+model_file[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            elif alg == 2 or alg == 3:
                filename = './result/time_res/'+model_file[alg]+str(shift+1)+'_'+str(slot)+'.pickle'
            else:
                filename = './result/time_res/'+model_file[alg]+'-'+str(shift+1)+'-'+str(slot+1)+'.pkl'

            if os.path.exists(filename):
                count += 1

                with open(filename,'rb') as f:
                    data = pickle.load(f)
                
                tmp += np.asarray(data,dtype=float)

    tmp /= count
    time_info[alg].extend(tmp/tmp[0])
# %%
