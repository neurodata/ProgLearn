#%%
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
import pandas as pd
from itertools import product
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib
#%%
def calc_forget(err, total_task, reps):
#Tom Vient et al
    forget = 0
    for ii in range(total_task-1):
        forget += err[ii][ii] - err[total_task-1][ii]

    forget /= (total_task-1)
    return forget/reps

def calc_transfer(err, single_err, total_task, reps):
#Tom Vient et al
    transfer = np.zeros(total_task,dtype=float)

    for ii in range(total_task):
        transfer[ii] = (single_err[ii] - err[total_task-1][ii])/reps

    return np.mean(transfer)

def calc_acc(err, total_task, reps):
#Tom Vient et al
    acc = 0
    for ii in range(total_task):
        acc += (1-err[total_task-1][ii]/reps)
    return acc/total_task

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def calc_avg_acc(err, total_task, reps):
    avg_acc = np.zeros(total_task, dtype=float)
    avg_var = np.zeros(total_task, dtype=float)
    for i in range(total_task):
        avg_acc[i] = (1*(i+1) - np.sum(err[i])/reps + (4-i)*.1)/10
        avg_var[i] = np.var(1-np.array(err[i])/reps)
    return avg_acc, avg_var

def calc_avg_single_acc(err, total_task, reps):
    avg_acc = np.zeros(total_task, dtype=float)
    avg_var = np.zeros(total_task, dtype=float)
    for i in range(total_task):
        avg_acc[i] = (1*(i+1) - np.sum(err[:i+1])/reps + (total_task-1-i)*.1)/10
        avg_var[i] = np.var(1-np.array(err[:i+1])/reps)
    return avg_acc, avg_var
    
def get_fte_bte(err, single_err, total_task):
    bte = [[] for i in range(total_task)]
    te = [[] for i in range(total_task)]
    fte = []
    
    for i in range(total_task):
        for j in range(i,total_task):
            #print(err[j][i],j,i)
            bte[i].append((err[i][i]+1e-2)/(err[j][i]+1e-2))
            te[i].append((single_err[i]+1e-2)/(err[j][i]+1e-2))
                
    for i in range(total_task):
        fte.append((single_err[i]+1e-2)/(err[i][i]+1e-2))
            
            
    return fte,bte,te

def calc_mean_bte(btes,task_num,reps=10):
    mean_bte = [[] for i in range(task_num)]


    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(btes[i][j])
        
        tmp=tmp/reps
        mean_bte[j].extend(tmp)
            
    return mean_bte     

def calc_mean_te(tes,task_num,reps=10):
    mean_te = [[] for i in range(task_num)]

    for j in range(task_num):
        tmp = 0
        for i in range(reps):
            tmp += np.array(tes[i][j])
        
        tmp=tmp/reps
        mean_te[j].extend(tmp)
                                             
    return mean_te 

def calc_mean_fte(ftes,task_num,reps=1):
    fte = np.asarray(ftes)
    
    return list(np.mean(np.asarray(fte),axis=0))

def get_error_matrix(filename, total_task):
    multitask_df, single_task_df = unpickle(filename)

    err = [[] for _ in range(total_task)]

    for ii in range(total_task):
        err[ii].extend(
            1 - np.array(
                multitask_df[multitask_df['base_task']==ii+1]['accuracy']
                )
            )
    single_err = 1 - np.array(single_task_df['accuracy'])

    return single_err, err

def sum_error_matrix(error_mat1, error_mat2, total_task):
    err = [[] for _ in range(total_task)]

    for ii in range(total_task):
        err[ii].extend(
            list(
                np.asarray(error_mat1[ii]) +
                np.asarray(error_mat2[ii])
            )
        )
    return err

def stratified_scatter(te_dict,axis_handle,s,color,style):
    algo = list(te_dict.keys())
    total_alg = len(algo)

    total_points = len(te_dict[algo[0]])

    pivot_points = np.arange(-.25, (total_alg+1)*1, step=1)
    interval = .7/(total_points-1)

    for algo_no,alg in enumerate(algo):
        for no,points in enumerate(te_dict[alg]):
            axis_handle.scatter(
                pivot_points[algo_no]+interval*no,
                te_dict[alg][no],
                s=s,
                c='k',
                marker=style[algo_no]
                )

#%%
btes_all = {}
ftes_all = {}
tes_all = {}
te_scatter = {}
df_all = {}

fle = {}
ble = {}
le = {}
#%%
### MAIN HYPERPARAMS ###
task_num = 5
total_alg = 13
combined_alg_name = ['SynN','SynF', 'Model Zoo', 'LwF','EWC','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'Total Replay', 'Partial Replay', 'None']
btes = [[] for i in range(total_alg)]
ftes = [[] for i in range(total_alg)]
tes = [[] for i in range(total_alg)]
model_file_combined = ['synn','synf', 'model_zoo', 'LwF', 'EWC', 'OEWC', 'si', 'er', 'agem', 'tag', 'offline', 'exact', 'None']
avg_acc = [[] for i in range(total_alg)]
avg_var = [[] for i in range(total_alg)]
avg_single_acc = [[] for i in range(total_alg)]
avg_single_var = [[] for i in range(total_alg)]
########################

#%% code for 500 samples
reps = 1

for alg in range(total_alg): 
    count = 0 

    for rep in range(reps):
        filename = '/Users/jayantadey/ProgLearn/benchmarks/five_datasets/results/'+model_file_combined[alg]+'.pickle'

        multitask_df, single_task_df = unpickle(filename)

        single_err_, err_ = get_error_matrix(filename, task_num)

        if count == 0:
            single_err, err = single_err_, err_
        else:
            err = sum_error_matrix(err, err_, total_task=task_num)
            single_err = list(
                np.asarray(single_err) + np.asarray(single_err_)
            )

        count += 1
    #single_err /= reps
    #err /= reps
    fte, bte, te = get_fte_bte(err,single_err, total_task=task_num)
    avg_acc_, avg_var_ = calc_avg_acc(err, task_num, reps)
    avg_single_acc_, avg_single_var_ = calc_avg_single_acc(single_err, task_num, reps)

    btes[alg].extend(bte)
    ftes[alg].extend(fte)
    tes[alg].extend(te)
    avg_acc[alg]= avg_acc_
    avg_var[alg] = avg_var_
    avg_single_acc[alg]= avg_single_acc_
    avg_single_var[alg] = avg_single_var_

    print('Algo name:' , combined_alg_name[alg])
    print('Accuracy', np.round(calc_acc(err, task_num, reps),2))
    print('forget', np.round(calc_forget(err, task_num, reps),2))
    print('transfer', np.round(calc_transfer(err, single_err, task_num, reps),2))

#%%
te = {'SynN':np.zeros(5,dtype=float), 'SynF':np.zeros(5,dtype=float), 'Model Zoo':np.zeros(5,dtype=float), 
    'LwF':np.zeros(5,dtype=float), 'EWC':np.zeros(5,dtype=float), 
    'O-EWC':np.zeros(5,dtype=float), 'SI':np.zeros(5,dtype=float),
    'ER':np.zeros(5,dtype=float), 'A-GEM':np.zeros(5,dtype=float),
    'TAG':np.zeros(5,dtype=float),
    'Total Replay':np.zeros(5,dtype=float), 'Partial Replay':np.zeros(5,dtype=float), 
    'None':np.zeros(5,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(5):
        te[name][i] = np.log(tes[count][i][4-i])

for name in te.keys():
    print(name, np.round(np.mean(te[name]),2), np.round(np.std(te[name], ddof=1),2))


df = pd.DataFrame.from_dict(te)
df = pd.melt(df,var_name='Algorithms', value_name='Transfer Efficieny')

#%%
fle['five_dataset'] = np.mean(np.log(ftes), axis=1)
ble['five_dataset'] = []
le['five_dataset'] = []

bte_end = {'SynN':np.zeros(5,dtype=float), 'SynF':np.zeros(5,dtype=float), 'Model Zoo':np.zeros(5,dtype=float), 
    'LwF':np.zeros(5,dtype=float), 'EWC':np.zeros(5,dtype=float), 
    'O-EWC':np.zeros(5,dtype=float), 'SI':np.zeros(5,dtype=float),
    'ER':np.zeros(5,dtype=float), 'A-GEM':np.zeros(5,dtype=float),
    'TAG':np.zeros(5,dtype=float),
    'Total Replay':np.zeros(5,dtype=float), 'Partial Replay':np.zeros(5,dtype=float), 
    'None':np.zeros(5,dtype=float)}
for count,name in enumerate(te.keys()):
    for i in range(5):
        bte_end[name][i] = np.log(btes[count][i][4-i])


for key in te.keys():
    le['five_dataset'].append(
        np.mean(te[key])
    )

for key in te.keys():
    ble['five_dataset'].append(
        np.mean(bte_end[key])
    )
#%%
btes_all['five_dataset'] = btes
ftes_all['five_dataset'] = ftes
df_all['five_dataset'] = df
tes_all['five_dataset'] = tes
te_scatter['five_dataset'] = te

#%%

### MAIN HYPERPARAMS ###
task_num = 20
total_alg = 13
combined_alg_name = ['SynN','SynF', 'Model Zoo', 'LwF','EWC','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'Total Replay', 'Partial Replay', 'None']
btes = [[] for i in range(total_alg)]
ftes = [[] for i in range(total_alg)]
tes = [[] for i in range(total_alg)]
model_file_combined = ['synn','synf', 'model_zoo', 'LwF', 'EWC', 'OEWC', 'si', 'er', 'agem', 'tag', 'offline', 'exact', 'None']
avg_acc = [[] for i in range(total_alg)]
avg_var = [[] for i in range(total_alg)]
avg_single_acc = [[] for i in range(total_alg)]
avg_single_var = [[] for i in range(total_alg)]
########################

#%% 
reps = 1

for alg in range(total_alg): 
    count = 0 

    for rep in range(reps):
        filename = '/Users/jayantadey/ProgLearn/benchmarks/mini_imagenet/results/'+model_file_combined[alg]+'.pickle'

        multitask_df, single_task_df = unpickle(filename)

        single_err_, err_ = get_error_matrix(filename, task_num)

        if count == 0:
            single_err, err = single_err_, err_
        else:
            err = sum_error_matrix(err, err_, task_num)
            single_err = list(
                np.asarray(single_err) + np.asarray(single_err_)
            )

        count += 1
    #single_err /= reps
    #err /= reps
    fte, bte, te = get_fte_bte(err,single_err, task_num)
    avg_acc_, avg_var_ = calc_avg_acc(err, task_num, reps)
    avg_single_acc_, avg_single_var_ = calc_avg_single_acc(single_err, task_num, reps)

    btes[alg].extend(bte)
    ftes[alg].extend(fte)
    tes[alg].extend(te)
    avg_acc[alg]= avg_acc_
    avg_var[alg] = avg_var_
    avg_single_acc[alg]= avg_single_acc_
    avg_single_var[alg] = avg_single_var_

    print('Algo name:' , combined_alg_name[alg])
    print('Accuracy', np.round(calc_acc(err, task_num, reps),2))
    print('forget', np.round(calc_forget(err, task_num, reps),2))
    print('transfer', np.round(calc_transfer(err, single_err, task_num, reps),2))

#%%
te = {'SynN':np.zeros(20,dtype=float), 'SynF':np.zeros(20,dtype=float), 'Model Zoo':np.zeros(20,dtype=float), 
    'LwF':np.zeros(20,dtype=float), 'EWC':np.zeros(20,dtype=float), 
    'O-EWC':np.zeros(20,dtype=float), 'SI':np.zeros(20,dtype=float),
    'ER':np.zeros(20,dtype=float), 'A-GEM':np.zeros(20,dtype=float),
    'TAG':np.zeros(20,dtype=float),
    'Total Replay':np.zeros(20,dtype=float), 'Partial Replay':np.zeros(20,dtype=float), 
    'None':np.zeros(20,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(20):
        te[name][i] = np.log(tes[count][i][19-i])


for name in te.keys():
    print(name, np.round(np.mean(te[name]),2), np.round(np.std(te[name], ddof=1),2))

df = pd.DataFrame.from_dict(te)
df = pd.melt(df,var_name='Algorithms', value_name='Transfer Efficieny')


#%%
fle['imagenet'] = np.mean(np.log(ftes), axis=1)
ble['imagenet'] = []
le['imagenet'] = []

bte_end = {'SynN':np.zeros(20,dtype=float), 'SynF':np.zeros(20,dtype=float), 'Model Zoo':np.zeros(20,dtype=float), 
    'LwF':np.zeros(20,dtype=float), 'EWC':np.zeros(20,dtype=float), 
    'O-EWC':np.zeros(20,dtype=float), 'SI':np.zeros(20,dtype=float),
    'ER':np.zeros(20,dtype=float), 'A-GEM':np.zeros(20,dtype=float),
    'TAG':np.zeros(20,dtype=float),
    'Total Replay':np.zeros(20,dtype=float), 'Partial Replay':np.zeros(20,dtype=float), 
    'None':np.zeros(20,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(20):
        bte_end[name][i] = np.log(btes[count][i][19-i])

for key in te.keys():
    le['imagenet'].append(
        np.mean(te[key])
    )

for key in te.keys():
    ble['imagenet'].append(
        np.mean(bte_end[key])
    )
#%%
btes_all['imagenet'] = btes
ftes_all['imagenet'] = ftes
df_all['imagenet'] = df
tes_all['imagenet'] = tes
te_scatter['imagenet'] = te
#%%

### MAIN HYPERPARAMS ###
task_num = 50
total_alg = 4
combined_alg_name = ['SynN','SynF', 'Model Zoo', 'LwF']
btes = [[] for i in range(total_alg)]
ftes = [[] for i in range(total_alg)]
tes = [[] for i in range(total_alg)]
model_file_combined = ['synn','synf', 'model_zoo', 'LwF']
avg_acc = [[] for i in range(total_alg)]
avg_var = [[] for i in range(total_alg)]
avg_single_acc = [[] for i in range(total_alg)]
avg_single_var = [[] for i in range(total_alg)]
########################

#%% 

for alg in range(total_alg): 

    filename = '/Users/jayantadey/ProgLearn/benchmarks/food1k/results/'+model_file_combined[alg]+'.pickle'

    multitask_df, single_task_df = unpickle(filename)

    single_err, err = get_error_matrix(filename, task_num)

    #single_err /= reps
    #err /= reps
    fte, bte, te = get_fte_bte(err,single_err, task_num)
    avg_acc_, avg_var_ = calc_avg_acc(err, task_num, 1)
    avg_single_acc_, avg_single_var_ = calc_avg_single_acc(single_err, task_num, 1)

    btes[alg].extend(bte)
    ftes[alg].extend(fte)
    tes[alg].extend(te)
    avg_acc[alg]= avg_acc_
    avg_var[alg] = avg_var_
    avg_single_acc[alg]= avg_single_acc_
    avg_single_var[alg] = avg_single_var_

    print('Algo name:' , combined_alg_name[alg])
    print('Accuracy', np.round(calc_acc(err, task_num, 1),2))
    print('forget', np.round(calc_forget(err, task_num, 1),2))
    print('transfer', np.round(calc_transfer(err, single_err, task_num, 1),2))

#%%
te = {'SynN':np.zeros(50,dtype=float), 'SynF':np.zeros(50,dtype=float), 'Model Zoo':np.zeros(50,dtype=float), 
    'LwF':np.zeros(50,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(50):
        te[name][i] = np.log(tes[count][i][49-i])


for name in te.keys():
    print(name, np.round(np.mean(te[name]),2), np.round(np.std(te[name], ddof=1),2))

df = pd.DataFrame.from_dict(te)
df = pd.melt(df,var_name='Algorithms', value_name='Transfer Efficieny')



#%%
fle['food1k'] = np.mean(np.log(ftes), axis=1)
ble['food1k'] = []
le['food1k'] = []

bte_end = {'SynN':np.zeros(50,dtype=float), 'SynF':np.zeros(50,dtype=float), 'Model Zoo':np.zeros(50,dtype=float), 
    'LwF':np.zeros(50,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(50):
        bte_end[name][i] = np.log(btes[count][i][49-i])


for key in te.keys():
    le['food1k'].append(
        np.mean(te[key])
    )

for key in te.keys():
    ble['food1k'].append(
        np.mean(bte_end[key])
    )
#%%
btes_all['food1k'] = btes
ftes_all['food1k'] = ftes
df_all['food1k'] = df
tes_all['food1k'] = tes
te_scatter['food1k'] = te



# %%
### MAIN HYPERPARAMS ###
ntrees = 10
slots = 10
task_num = 10
shifts = 6
total_alg_top = 9
total_alg_bottom = 8
alg_name_top = ['SynN','SynF', 'Model Zoo','ProgNN', 'LMC', 'DF-CNN', 'EWC', 'Total Replay', 'Partial Replay']
alg_name_bottom = ['SynF','LwF','O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'None']
combined_alg_name = ['SynN','SynF', 'Model Zoo','ProgNN', 'DF-CNN','EWC', 'Total Replay', 'Partial Replay', 'LwF', 'O-EWC','SI', 'ER', 'A-GEM', 'TAG', 'None']
model_file_top = ['dnn0withrep','fixed_uf10withrep', 'model_zoo','Prog_NN', 'LMC', 'DF_CNN', 'EWC', 'offline', 'exact']
model_file_bottom = ['uf10withrep', 'LwF', 'OEWC', 'si', 'er', 'agem', 'tag', 'None']
btes_top = [[] for i in range(total_alg_top)]
ftes_top = [[] for i in range(total_alg_top)]
tes_top = [[] for i in range(total_alg_top)]
btes_bottom = [[] for i in range(total_alg_bottom)]
ftes_bottom = [[] for i in range(total_alg_bottom)]
tes_bottom = [[] for i in range(total_alg_bottom)]
avg_acc_top = [[] for i in range(total_alg_top)]
avg_var_top = [[] for i in range(total_alg_top)]
avg_acc_bottom = [[] for i in range(total_alg_bottom)]
avg_var_bottom = [[] for i in range(total_alg_bottom)]

avg_single_acc_top = [[] for i in range(total_alg_top)]
avg_single_var_top = [[] for i in range(total_alg_top)]
avg_single_acc_bottom = [[] for i in range(total_alg_bottom)]
avg_single_var_bottom = [[] for i in range(total_alg_bottom)]

#combined_alg_name = ['L2N','L2F','Prog-NN', 'DF-CNN','LwF','EWC','O-EWC','SI', 'Replay (increasing amount)', 'Replay (fixed amount)', 'None']
model_file_combined = ['dnn0withrep','fixed_uf10withrep', 'model_zoo','Prog_NN','DF_CNN', 'LwF', 'EWC', 'OEWC', 'si', 'offline', 'exact', 'er', 'agem', 'tag', 'None']

########################

#%% code for 500 samples
reps = slots*shifts

for alg in range(total_alg_top): 
    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)]

    for slot in range(slots):
        for shift in range(shifts):
            if alg < 2:
                filename = '/Users/jayantadey/ProgLearn/benchmarks/cifar_exp/result/result/'+model_file_top[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            elif alg == 3 or alg == 5:
                filename = '/Users/jayantadey/ProgLearn/benchmarks/cifar_exp/benchmarking_algorthms_result/'+model_file_top[alg]+'-'+str(shift+1)+'-'+str(slot+1)+'.pickle'
            elif alg == 2:
                filename = '/Users/jayantadey/ProgLearn/benchmarks/cifar_exp/benchmarking_algorthms_result/'+model_file_top[alg]+'_'+str(slot+1)+'_'+str(shift+1)+'.pickle'
            else:
                filename = '/Users/jayantadey/ProgLearn/benchmarks/cifar_exp/benchmarking_algorthms_result/'+model_file_top[alg]+'-'+str(slot+1)+'-'+str(shift+1)+'.pickle'

            multitask_df, single_task_df = unpickle(filename)

            single_err_, err_ = get_error_matrix(filename, task_num)

            if count == 0:
                single_err, err = single_err_, err_
            else:
                err = sum_error_matrix(err, err_, task_num)
                single_err = list(
                    np.asarray(single_err) + np.asarray(single_err_)
                )

            count += 1
    #single_err /= reps
    #err /= reps
    fte, bte, te = get_fte_bte(err,single_err,task_num)
    avg_acc, avg_var = calc_avg_acc(err, task_num, reps)
    avg_single_acc, avg_single_var = calc_avg_single_acc(single_err, task_num, reps)

    btes_top[alg].extend(bte)
    ftes_top[alg].extend(fte)
    tes_top[alg].extend(te)
    avg_acc_top[alg]= avg_acc
    avg_var_top[alg] = avg_var
    avg_single_acc_top[alg]= avg_single_acc
    avg_single_var_top[alg] = avg_single_var

    print('Algo name:' , alg_name_top[alg])
    print('Accuracy', np.round(calc_acc(err,task_num, reps),2))
    print('forget', np.round(calc_forget(err,task_num, reps),2))
    print('transfer', np.round(calc_transfer(err, single_err,task_num, reps),2))
    

# %%
reps = slots*shifts

for alg in range(total_alg_bottom): 
    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)]

    for slot in range(slots):
        for shift in range(shifts):
            if alg < 1:
                filename = '/Users/jayantadey/ProgLearn/benchmarks/cifar_exp/result/result/'+model_file_bottom[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            else:
                filename = '/Users/jayantadey/ProgLearn/benchmarks/cifar_exp/benchmarking_algorthms_result/'+model_file_bottom[alg]+'-'+str(slot+1)+'-'+str(shift+1)+'.pickle'

            multitask_df, single_task_df = unpickle(filename)

            single_err_, err_ = get_error_matrix(filename, task_num)

            if count == 0:
                single_err, err = single_err_, err_
            else:
                err = sum_error_matrix(err, err_, task_num)
                single_err = list(
                    np.asarray(single_err) + np.asarray(single_err_)
                )

            count += 1
    #single_err /= reps
    #err /= reps
    fte, bte, te = get_fte_bte(err,single_err,task_num)
    avg_acc, avg_var = calc_avg_acc(err, task_num, reps)
    avg_single_acc, avg_single_var = calc_avg_single_acc(single_err, task_num, reps)


    btes_bottom[alg].extend(bte)
    ftes_bottom[alg].extend(fte)
    tes_bottom[alg].extend(te)
    avg_acc_bottom[alg] = avg_acc
    avg_var_bottom[alg] = avg_var
    avg_single_acc_bottom[alg]= avg_single_acc
    avg_single_var_bottom[alg] = avg_single_var

    print('Algo name:' , alg_name_bottom[alg])
    print('Accuracy', np.round(calc_acc(err,task_num, reps),2))
    print('forget', np.round(calc_forget(err,task_num, reps),2))
    print('transfer', np.round(calc_transfer(err, single_err,task_num, reps),2))
#%%
te_500 = {'SynN':np.zeros(10,dtype=float), 'SynF':np.zeros(10,dtype=float), 
          'Model Zoo':np.zeros(10,dtype=float),
          'ProgNN':np.zeros(10,dtype=float), 'LMC':np.zeros(10,dtype=float),
          'DF-CNN':np.zeros(10,dtype=float), 
          'EWC':np.zeros(10,dtype=float),'Total Replay':np.zeros(10,dtype=float),
          'Partial Replay':np.zeros(10,dtype=float),
          'SynF (constrained)':np.zeros(10,dtype=float), 'LwF':np.zeros(10,dtype=float),
           'O-EWC':np.zeros(10,dtype=float), 'SI':np.zeros(10,dtype=float),
          'ER':np.zeros(10,dtype=float), 'A-GEM':np.zeros(10,dtype=float),
          'TAG':np.zeros(10,dtype=float), 'None':np.zeros(10,dtype=float)}

for count,name in enumerate(te_500.keys()):
    #print(name, count)
    for i in range(10):
        if count <9:
            te_500[name][i] = np.log(tes_top[count][i][9-i])
        else:
            te_500[name][i] = np.log(tes_bottom[count-9][i][9-i])


for name in te_500.keys():
    print(name, np.round(np.mean(te_500[name]),2), np.round(np.std(te_500[name], ddof=1),2))

df_500 = pd.DataFrame.from_dict(te_500)
df_500 = pd.melt(df_500,var_name='Algorithms', value_name='Learning Efficieny')

# %%
fle['cifar'] = np.concatenate((
    np.mean(np.log(ftes_top), axis=1),
    np.mean(np.log(ftes_bottom), axis=1)
))
ble['cifar'] = []
le['cifar'] = []

bte_end = {'SynN':np.zeros(10,dtype=float), 'SynF':np.zeros(10,dtype=float), 
          'Model Zoo':np.zeros(10,dtype=float),
          'ProgNN':np.zeros(10,dtype=float), 'LMC':np.zeros(10,dtype=float),
          'DF-CNN':np.zeros(10,dtype=float), 
          'EWC':np.zeros(10,dtype=float),'Total Replay':np.zeros(10,dtype=float),
          'Partial Replay':np.zeros(10,dtype=float),
          'SynF (constrained)':np.zeros(10,dtype=float), 'LwF':np.zeros(10,dtype=float),
           'O-EWC':np.zeros(10,dtype=float), 'SI':np.zeros(10,dtype=float),
          'ER':np.zeros(10,dtype=float), 'A-GEM':np.zeros(10,dtype=float),
          'TAG':np.zeros(10,dtype=float), 'None':np.zeros(10,dtype=float)}


for count,name in enumerate(bte_end.keys()):
    #print(name, count)
    for i in range(10):
        if count <9:
            bte_end[name][i] = np.log(btes_top[count][i][9-i])
        else:
            bte_end[name][i] = np.log(btes_bottom[count-9][i][9-i])


for key in te_500.keys():
    le['cifar'].append(
        np.mean(te_500[key])
    )

for key in bte_end.keys():
    ble['cifar'].append(
        np.mean(bte_end[key])
    )


#%%
btes_all['cifar'] = btes
ftes_all['cifar'] = ftes
df_all['cifar'] = df_500
tes_all['cifar'] = tes
te_scatter['cifar'] = te_500
# %%
### MAIN HYPERPARAMS ###
task_num = 6
total_alg = 10
combined_alg_name = ['SynN','SynF', 'Model Zoo','LwF','EWC','O-EWC','SI', 'Total Replay', 'Partial Replay', 'None']
btes = [[] for i in range(total_alg)]
ftes = [[] for i in range(total_alg)]
tes = [[] for i in range(total_alg)]
model_file_combined = ['odin','odif', 'model_zoo', 'LwF', 'EWC', 'OEWC', 'si', 'offline', 'exact', 'None']
avg_acc = [[] for i in range(total_alg)]
avg_var = [[] for i in range(total_alg)]
avg_single_acc = [[] for i in range(total_alg)]
avg_single_var = [[] for i in range(total_alg)]
########################

#%% code for 500 samples
reps = 10

for alg in range(total_alg): 
    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)]

    for rep in range(reps):
        filename = '/Users/jayantadey/ProgLearn/benchmarks/spoken_digit/'+model_file_combined[alg]+'-'+str(rep)+'.pickle'

        multitask_df, single_task_df = unpickle(filename)

        single_err_, err_ = get_error_matrix(filename, task_num)

        if count == 0:
            single_err, err = single_err_, err_
        else:
            err = sum_error_matrix(err, err_, task_num)
            single_err = list(
                np.asarray(single_err) + np.asarray(single_err_)
            )

        count += 1
    #single_err /= reps
    #err /= reps
    fte, bte, te = get_fte_bte(err,single_err,task_num)

    avg_acc_, avg_var_ = calc_avg_acc(err, task_num, reps)
    avg_single_acc_, avg_single_var_ = calc_avg_single_acc(single_err, task_num, reps)

    btes[alg].extend(bte)
    ftes[alg].extend(fte)
    tes[alg].extend(te)
    avg_acc[alg]= avg_acc_
    avg_var[alg] = avg_var_
    avg_single_acc[alg]= avg_single_acc_
    avg_single_var[alg] = avg_single_var_

    print('Algo name:' , combined_alg_name[alg])
    print('Accuracy', np.round(calc_acc(err, task_num, reps),2))
    print('forget', np.round(calc_forget(err, task_num, reps),2))
    print('transfer', np.round(calc_transfer(err, single_err, task_num, reps),2))

#%%
te = {'SynN':np.zeros(6,dtype=float), 'SynF':np.zeros(6,dtype=float), 
    'Model Zoo':np.zeros(6,dtype=float), 'LwF':np.zeros(6,dtype=float), 
    'EWC':np.zeros(6,dtype=float), 
    'O-EWC':np.zeros(6,dtype=float), 'SI':np.zeros(6,dtype=float),
    'Total Replay':np.zeros(6,dtype=float), 'Partial Replay':np.zeros(6,dtype=float), 
    'None':np.zeros(6,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(6):
        te[name][i] = np.log(tes[count][i][5-i])


df = pd.DataFrame.from_dict(te)
df = pd.melt(df,var_name='Algorithms', value_name='Transfer Efficieny')

# %%
fle['spoken'] = np.mean(np.log(ftes), axis=1)
ble['spoken'] = []
le['spoken'] = []

bte_end = {'SynN':np.zeros(6,dtype=float), 'SynF':np.zeros(6,dtype=float), 
    'Model Zoo':np.zeros(6,dtype=float), 'LwF':np.zeros(6,dtype=float), 
    'EWC':np.zeros(6,dtype=float), 
    'O-EWC':np.zeros(6,dtype=float), 'SI':np.zeros(6,dtype=float),
    'Total Replay':np.zeros(6,dtype=float), 'Partial Replay':np.zeros(6,dtype=float), 
    'None':np.zeros(6,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(6):
        bte_end[name][i] = np.log(btes[count][i][5-i])


for key in te.keys():
    le['spoken'].append(
        np.mean(te[key])
    )

for key in te.keys():
    ble['spoken'].append(
        np.mean(bte_end[key])
    )
#%%
btes_all['spoken'] = btes
ftes_all['spoken'] = ftes
df_all['spoken'] = df
tes_all['spoken'] = tes
te_scatter['spoken'] = te
# %%
fle_sorted = {}
ble_sorted = {}
le_sorted = {}
data = {}

sorting_idx = [ 0,  2,  1,  3,  9,  4,  5, 13,  7,  8, 14, 15, 10, 11, 12,  6, 16]
algorithms_ours = []
algorithms_ = list(te_scatter['cifar'].keys())

for idx in sorting_idx:
    algorithms_ours.append(algorithms_[idx])

keys = ['forward learning', 'backward learning', 'overall learning']

for data_key in fle.keys():
    fle_sorted[data_key] = []
    ble_sorted[data_key] = []
    le_sorted[data_key] = []

    alg_ = list(te_scatter[data_key].keys())
    #print(alg_)
    
    for alg in algorithms_ours:
       # print(alg)
        
        if alg in alg_:
            fle_sorted[data_key].append(
                fle[data_key][alg_.index(alg)]
            )
            ble_sorted[data_key].append(
                ble[data_key][alg_.index(alg)]
            )
            le_sorted[data_key].append(
                le[data_key][alg_.index(alg)]
            )
        else:
            fle_sorted[data_key].append(
                np.nan
            )
            ble_sorted[data_key].append(
                np.nan
            )
            le_sorted[data_key].append(
                np.nan
            )

#%%
data['forward learning'] = pd.DataFrame({"cifar":fle_sorted["cifar"], "spoken":fle_sorted["spoken"], "food1k":fle_sorted["food1k"],\
        "imagenet":fle_sorted["imagenet"], "5-dataset":fle_sorted["five_dataset"]})
data['forward learning'].index = algorithms_ours

data['forward learning']['mean'] = data['forward learning'].mean(axis=1)

data['backward learning'] = pd.DataFrame({"cifar":ble_sorted["cifar"], "spoken":ble_sorted["spoken"], "food1k":ble_sorted["food1k"],\
    "imagenet":ble_sorted["imagenet"], "5-dataset":ble_sorted["five_dataset"]})
data['backward learning'].index = algorithms_ours

data['backward learning']['mean'] = data['backward learning'].mean(axis=1)

data['overall learning'] = pd.DataFrame({"cifar":le_sorted["cifar"], "spoken":le_sorted["spoken"], "food1k":le_sorted["food1k"],\
        "imagenet":le_sorted["imagenet"], "5-dataset":le_sorted["five_dataset"]})
data['overall learning'].index = algorithms_ours


data['overall learning']['mean'] = data['overall learning'].mean(axis=1)
mean_le = list(data['overall learning']['mean'])
print(np.argsort(mean_le)[::-1])

#%%
accuracy = {}
accuracy_ = {}
forget = {}
forget_ = {}
transfer_ = {}
transfer = {}
data_ven = {}
algorithms_ = ['SynN', 'SynF', 'ProgNN', 'LMC', \
            'DF-CNN', 'EWC', 'Total Replay', \
            'Partial Replay', 'Model Zoo', \
            'SynF (constrained)', \
            'LwF', 'O-EWC', 'SI', 'ER', \
            'A-GEM', 'TAG', 'None']
algorithms = []

sorted_indx = [ 0,  2,  8,  1,  9,  3,  7,  6, 14, 10, 13,  4, 15, 11, 12,  5, 16]

accuracy_['cifar'] = [.4, .41, .39, 0.28, .17, .36, .36, .37, .46,\
           .41, .42, .36, .35, .32, .27, .15, .29]
forget_['cifar'] = [.03, .03, 0, -0.04, -.09, -.01, -.03, -.01, .05,\
          .03, 0, 0, -.01, -.13, -.17, -.05, -.14]
transfer_['cifar'] = [.13, .08, .07,  -0.04, -.09, -.09, -.09, -.07,\
            .05, .03, -.03, -.08, -.09, -.09, -.13,\
            -.23, -.16]

accuracy_['speech'] = [.91, .91, np.nan, np.nan, np.nan, .73, .93, .93, .98,\
            np.nan, .76, .72, .72, np.nan, np.nan, np.nan, .73]
forget_['speech'] = [.035, .01, np.nan, np.nan, np.nan, -.28, 0, 0, 0.01,\
            np.nan, -.24, -.29, -.27, np.nan, np.nan, np.nan, -.3]
transfer_['speech'] = [.16, .03, np.nan, np.nan, np.nan, -.24, -.04, -.04, 0,\
            np.nan, -.21, -.25, -.25, np.nan, np.nan, np.nan, -.23]

accuracy_['food1k'] = [.45, .36, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, .42,\
           np.nan, .43, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
forget_['food1k'] = [.03, .01, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, .05,\
           np.nan, -.08, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
transfer_['food1k'] = [.2, .09, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, .08,\
           np.nan, -.03, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]


accuracy_['imagenet'] = [.55, .52, np.nan, np.nan, np.nan, .54, .58, .58, .6,\
            np.nan, .6, .55, .57, .58, .56, .58, .52]
forget_['imagenet'] = [.03, .02, np.nan, np.nan, np.nan, -.04, -.01, -.01, .06,\
            np.nan, 0, -.03, -.02, -.07, -.07, -.06, -.12]
transfer_['imagenet'] = [.02, .04, np.nan, np.nan, np.nan, -.07, -.03, -.03, .1,\
            np.nan, -.01, -.06, -.04, -.01, .05, -.04, -.1]

accuracy_['5 dataset'] = [.79, .71, np.nan, np.nan, np.nan, .68, .83, .82, .89,\
            np.nan, .8, .68, .66, .68, .67, .69, .62]
forget_['5 dataset'] = [-.02, -.01, np.nan, np.nan, np.nan, -.08, -.01, -.01, .03,\
            np.nan, -.07, -.08, -.07, -.13, -.14, -.11, -.31]
transfer_['5 dataset'] = [-.03, -.02, np.nan, np.nan, np.nan, -.18, -.03, -.04, .03,\
            np.nan, -.06, -.18, -.2, -.12, -.09, -.1, -.24]


for idx in sorted_indx:
    algorithms.append(algorithms_[idx])

for key in transfer_.keys():
    accuracy[key] = []
    forget[key] = []
    transfer[key] = []
    for idx in sorted_indx:
        accuracy[key].append(accuracy_[key][idx])
        forget[key].append(forget_[key][idx])
        transfer[key].append(transfer_[key][idx])

keys = ['accuracy', 'forget', 'transfer']

data_ven['accuracy'] = pd.DataFrame({"cifar":accuracy["cifar"], "speech":accuracy["speech"], "food1k":accuracy["food1k"],\
                    "imagenet":accuracy["imagenet"], "5-dataset":accuracy["5 dataset"]})
data_ven['accuracy'].index = algorithms
data_ven['accuracy']['mean'] = data_ven['accuracy'].mean(axis=1)


data_ven['forget'] = pd.DataFrame({"cifar":forget["cifar"], "speech":forget["speech"],\
                    "food1k":forget["food1k"], "imagenet":forget["imagenet"], "5-dataset":forget["5 dataset"]})
data_ven['forget'].index = algorithms
data_ven['forget']['mean'] = data_ven['forget'].mean(axis=1)


data_ven['transfer'] = pd.DataFrame({"cifar":transfer["cifar"], "speech":transfer["speech"], "food1k":transfer["food1k"],\
                    "imagenet":transfer["imagenet"], "5-dataset":transfer["5 dataset"]})
data_ven['transfer'].index = algorithms
data_ven['transfer']['mean'] = data_ven['transfer'].mean(axis=1)
mean_trn = list(data_ven['transfer']['mean'])
print(np.argsort(mean_trn)[::-1])

#%%
keys_ours = ['forward learning', 'backward learning', 'overall learning']
keys_ =  ['Forward', 'Backward', 'Overall']
fig, ax = plt.subplots(2, 3, figsize=(18,16), sharex=True)
sns.set_context('talk')
vmins = [-1.5,-1.5,-1.5]
vmaxs = [1.5,1.5,1.5]
clr = ["#984ea3","#984ea3","#984ea3","#984ea3","#4daf4a","#984ea3","#984ea3","#4daf4a","#4daf4a","#4daf4a","#4daf4a","#4daf4a","#4daf4a","#4daf4a","#4daf4a","#4daf4a","#4daf4a"]
for i in range(3):
    sns.heatmap(data[keys_ours[i]], yticklabels=algorithms_ours,\
                vmin=vmins[i], vmax=vmaxs[i],
             cmap='coolwarm', ax=ax[0][i],)# cbar=i == 0, \
             #cbar_ax=cbar_ax if i==0 else None)

    if i != 0:
        ax[0][i].set_yticks([])

    ax[0][i].set_title(keys_[i], fontsize=35)
    ax[0][i].set_xticklabels(ax[0][i].get_xticklabels(), rotation=70, fontsize=22)
    #ax[i].set_yticklabels(ax[i].get_yticklabels(), fontsize=20)

for ytick, color in zip(ax[0][0].get_yticklabels(), clr):
    ytick.set_color(color)
    ytick.set_fontsize(18)



### veniat's ###
clr = ["#984ea3","#984ea3","#984ea3","#984ea3","#4daf4a","#984ea3","#4daf4a","#4daf4a","#4daf4a","#4daf4a","#4daf4a","#984ea3","#4daf4a","#4daf4a","#4daf4a","#4daf4a","#4daf4a"]

keys = ['accuracy', 'forget', 'transfer']
keys_ = ['Accuracy', 'Forget', 'Transfer']
vmins = [0,-.3,-.3]
vmaxs = [1,.3,.3]

for i in range(3):
    sns.heatmap(data_ven[keys[i]], yticklabels=algorithms,\
                vmin=vmins[i], vmax=vmaxs[i],
             cmap='coolwarm', ax=ax[1][i],)# cbar=i == 0, \
             #cbar_ax=cbar_ax if i==0 else None)

    if i != 0:
        ax[1][i].set_yticks([])

    ax[1][i].set_title(keys_[i], fontsize=35)
    ax[1][i].set_xticklabels(ax[1][i].get_xticklabels(), rotation=70, fontsize=20)
    #ax[i].set_yticklabels(ax[i].get_yticklabels(), fontsize=20)

for ytick, color in zip(ax[1][0].get_yticklabels(), clr):
    ytick.set_color(color)


fig.tight_layout(rect=[0, 0, .9, 1])
plt.savefig('heatmap_performance_overall.pdf')
# %%
