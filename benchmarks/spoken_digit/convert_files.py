#%%
import numpy as np
import pandas as pd
import pickle
# %%
res = np.load('/Users/jayantadey/Downloads/cache_acc_tree00_dnn_repetition200.npy')

for fold in range(10):
    multitask_df = pd.DataFrame() 
    singletask_df = pd.DataFrame()

    accuracy_all = res[fold]
    num_tasks = 6
    datafold = []
    acc = []
    task = []
    base_task = []

    for i in range(num_tasks):
        for j in range(i+1):
            datafold.append(fold)
            task.append(j+1)
            base_task.append(i+1)
            acc.append(accuracy_all[7 * j + 1 + i]) 

    multitask_df['data_fold'] = datafold
    multitask_df['task'] = task
    multitask_df['base_task'] = base_task
    multitask_df['accuracy'] = acc

    task = []
    datafold = []
    acc = []

    for i in range(num_tasks):
        task.append(i)
        datafold.append(fold)
        acc.append(accuracy_all[7*i])

    singletask_df['task'] = task
    singletask_df['data_fold'] = datafold
    singletask_df['accuracy'] = acc

    df = (multitask_df,singletask_df)
    with open('odin-'+str(fold)+'.pickle','wb') as f:
        pickle.dump(df,f)
# %%
