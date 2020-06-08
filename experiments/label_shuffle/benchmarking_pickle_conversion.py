#%%
from itertools import product
import pandas as pd
import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
#%% code for EWC, Online_EWC, SI, LwF
algs = ["EWC", "Online_EWC", "SI", "LwF"]
shift_fold = 6
total_tasks = 10
slot_fold = 10

for alg in algs:
    for shift in range(shift_fold):
        # find multitask accuracies
        for slot in range(slot_fold):
            df = pd.DataFrame()
            shifts = []
            base_tasks=[]
            accuracies_across_tasks = []

            filename = '/data/Jayanta/final_LL_res/random_label_res/'+alg+str(shift+1)+'_'+str(slot)+'.pickle'
            accuracy = 1 - np.asarray(unpickle(filename))
            for base_task in range(total_tasks):
                shifts.append(shift+1)
                base_tasks.append(base_task+1)
                accuracies_across_tasks.append(accuracy[base_task][0])

            df['data_fold'] = shifts
            df['task'] = base_tasks
            df['task_1_accuracy'] = accuracies_across_tasks

            file_to_save = 'benchmarking_algorthms_result/'+alg+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            with open(file_to_save, 'wb') as f:
                pickle.dump(df, f)
#%% code for Prog-NN, DF-CNN
algs = ["Prog_NN", "DF_CNN"]
shift_fold = 6
total_tasks = 10
epoch_indx = list(range(100,1100,100))

for alg in algs:
    for shift in range(shift_fold):
        for slot in range(slot_fold):
            df = pd.DataFrame()
            shifts = []
            base_tasks=[]
            accuracies_across_tasks = []

            # find multitask accuracies
            filename = '/data/Jayanta/final_LL_res/slot_res/'+alg+str(shift+1)+'_'+str(slot)+'.pickle'
            accuracy = np.asarray(unpickle(filename))
            accuracy = accuracy[epoch_indx,:]
            for base_task in range(total_tasks):
                shifts.append(shift+1)
                base_tasks.append(base_task+1)
                accuracies_across_tasks.append(accuracy[base_task,0])

            df['data_fold'] = shifts
            df['task'] = base_tasks
            df['task_1_accuracy'] = accuracies_across_tasks

            file_to_save = 'benchmarking_algorthms_result/'+alg+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            with open(file_to_save, 'wb') as f:
                pickle.dump(df, f)
#%%
