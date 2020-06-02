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

for alg in algs:
    for shift in range(shift_fold):
        df = pd.DataFrame()
        shifts = []
        tasks = []
        base_tasks = []
        accuracies_across_tasks = []
        single_task_accuracies = np.zeros(10,dtype=float)

        # find multitask accuracies
        for base_task in range(total_tasks):
            filename = '/data/Jayanta/continual-learning/crossval_res/'+alg+str(base_task+1)+'__'+str(shift+1)+'.pickle'
            accuracy = 1 - np.asarray(unpickle(filename))

            for task in range(base_task+1):
                shifts.append(shift)
                tasks.append(task+1)
                base_tasks.append(base_task+1)
                accuracies_across_tasks.append(accuracy[task])

        df['data_fold'] = shifts
        df['task'] = tasks
        df['base_task'] = base_tasks
        df['accuracy'] = accuracies_across_tasks

        #find single task accuracies
        for task in range(total_tasks):
            filename = '/data/Jayanta/continual-learning/crossval_res_singletask/single_task'+str(task+1)+'__'+str(shift+1)+'.pickle'
            single_task_accuracies[task] = 1 - np.asarray(unpickle(filename))

        df_single_task = pd.DataFrame()
        df_single_task['task'] = range(1, 11)
        df_single_task['data_fold'] = shift+1
        df_single_task['accuracy'] = single_task_accuracies

        summary = (df,df_single_task)
        file_to_save = 'benchmarking_algorthms_result/'+alg+'_'+str(shift+1)+'.pickle'
        with open(file_to_save, 'wb') as f:
            pickle.dump(summary, f)
#%% code for Prog-NN, DF-CNN
algs = ["Prog_NN", "DF_CNN"]
shift_fold = 6
total_tasks = 10
epoch_indx = list(range(100,1100,100))

for alg in algs:
    for shift in range(shift_fold):
        df = pd.DataFrame()
        shifts = []
        tasks = []
        base_tasks = []
        accuracies_across_tasks = []
        single_task_accuracies = np.zeros(10,dtype=float)

        # find multitask accuracies
        for base_task in range(total_tasks):
            filename = '/data/Jayanta/continual-learning/crossval_res/'+alg+'__'+str(shift+1)+'.pickle'
            accuracy = np.asarray(unpickle(filename))
            accuracy = accuracy[epoch_indx,:]

            for task in range(base_task+1):
                shifts.append(shift+1)
                tasks.append(task+1)
                base_tasks.append(base_task+1)
                accuracies_across_tasks.append(accuracy[base_task,task])

        df['data_fold'] = shifts
        df['task'] = tasks
        df['base_task'] = base_tasks
        df['accuracy'] = accuracies_across_tasks

        #find single task accuracies
        for task in range(total_tasks):
            filename = '/data/Jayanta/continual-learning/crossval_res_singletask/'+alg+str(task+1)+'__'+str(shift+1)+'.pickle'
            single_task_accuracies[task] = unpickle(filename)[99][0]

        df_single_task = pd.DataFrame()
        df_single_task['task'] = range(1, 11)
        df_single_task['data_fold'] = shift
        df_single_task['accuracy'] = single_task_accuracies

        summary = (df,df_single_task)
        file_to_save = 'benchmarking_algorthms_result/'+alg+'_'+str(shift+1)+'.pickle'
        with open(file_to_save, 'wb') as f:
            pickle.dump(summary, f)
#%%
