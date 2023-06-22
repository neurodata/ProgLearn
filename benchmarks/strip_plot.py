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
labels = []
btes_all = {}
ftes_all = {}
tes_all = {}
te_scatter = {}
df_all = {}

fle = {}
ble = {}
le = {}
ordr = []
# %%
### MAIN HYPERPARAMS ###
ntrees = 10
slots = 10
task_num = 10
shifts = 6
total_alg_top = 8
total_alg_bottom = 4
alg_name_top = ['SynN','SynF', 'Model Zoo','ProgNN', 'LMC', 'DF-CNN', 'Total Replay', 'Partial Replay']
alg_name_bottom = ['SynF','LwF', 'A-GEM', 'None']
combined_alg_name = ['SynN','SynF', 'Model Zoo','ProgNN', 'LMC', 'DF-CNN', 'Total Replay', 'Partial Replay', 'LwF', 'A-GEM', 'None']

model_file_top = ['dnn0withrep','fixed_uf10withrep', 'model_zoo','Prog_NN', 'LMC', 'DF_CNN', 'offline', 'exact']
model_file_bottom = ['uf10withrep', 'LwF', 'agem', 'None']
btes_top = [[] for i in range(total_alg_top)]
ftes_top = [[] for i in range(total_alg_top)]
tes_top = [[] for i in range(total_alg_top)]
btes_bottom = [[] for i in range(total_alg_bottom)]
ftes_bottom = [[] for i in range(total_alg_bottom)]
tes_bottom = [[] for i in range(total_alg_bottom)]

#combined_alg_name = ['L2N','L2F','Prog-NN', 'DF-CNN','LwF','EWC','O-EWC','SI', 'Replay (increasing amount)', 'Replay (fixed amount)', 'None']
model_file_combined = ['dnn0withrep','fixed_uf10withrep', 'model_zoo','Prog_NN','DF_CNN', 'LwF', 'offline', 'exact', 'agem', 'None']

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
                filename = 'cifar_exp/result/result/'+model_file_top[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            elif alg == 3 or alg == 5:
                filename = 'cifar_exp/benchmarking_algorthms_result/'+model_file_top[alg]+'-'+str(shift+1)+'-'+str(slot+1)+'.pickle'
            elif alg == 2:
                filename = 'cifar_exp/benchmarking_algorthms_result/'+model_file_top[alg]+'_'+str(slot+1)+'_'+str(shift+1)+'.pickle'
            else:
                filename = 'cifar_exp/benchmarking_algorthms_result/'+model_file_top[alg]+'-'+str(slot+1)+'-'+str(shift+1)+'.pickle'

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
                filename = 'cifar_exp/result/result/'+model_file_bottom[alg]+'_'+str(shift+1)+'_'+str(slot)+'.pickle'
            else:
                filename = 'cifar_exp/benchmarking_algorthms_result/'+model_file_bottom[alg]+'-'+str(slot+1)+'-'+str(shift+1)+'.pickle'

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
    
#%%
te_500 = {'SynN*':np.zeros(10,dtype=float), 'SynF*':np.zeros(10,dtype=float), 
          'Model Zoo*':np.zeros(10,dtype=float),
          'ProgNN*':np.zeros(10,dtype=float), 'LMC*':np.zeros(10,dtype=float),
          'DF-CNN*':np.zeros(10,dtype=float),'Total Replay':np.zeros(10,dtype=float),
          'Partial Replay':np.zeros(10,dtype=float),
          'SynF (constrained)':np.zeros(10,dtype=float), 'LwF':np.zeros(10,dtype=float),
          'A-GEM':np.zeros(10,dtype=float), 'None':np.zeros(10,dtype=float)}


for count,name in enumerate(te_500.keys()):
    #print(name, count)
    for i in range(10):
        if count <8:
            te_500[name][i] = np.log(tes_top[count][i][9-i])
        else:
            te_500[name][i] = np.log(tes_bottom[count-8][i][9-i])


mean_val = []
for name in te_500.keys():
    mean_val.append(np.mean(te_500[name]))
    print(name, np.round(np.mean(te_500[name]),2), np.round(np.std(te_500[name], ddof=1),2))

arg = np.argsort(mean_val)[::-1]
ordr.append(arg)
algos = list(te_500.keys())
combined_alg_name = []

for ii in arg:
    combined_alg_name.append(
        algos[ii]
    )
    
tmp_te = {}
for id in combined_alg_name:
    tmp_te[id] = te_500[id]

df_le = pd.DataFrame.from_dict(tmp_te)
df_le = pd.melt(df_le,var_name='Algorithms', value_name='Transfer Efficieny')

# %%
fle['cifar'] = np.concatenate((
    np.mean(np.log(ftes_top), axis=1),
    np.mean(np.log(ftes_bottom), axis=1)
))
ble['cifar'] = []
le['cifar'] = []

bte_end = {'SynN*':np.zeros(10,dtype=float), 'SynF*':np.zeros(10,dtype=float), 
          'Model Zoo*':np.zeros(10,dtype=float),
          'ProgNN*':np.zeros(10,dtype=float), 'LMC*':np.zeros(10,dtype=float),
          'DF-CNN*':np.zeros(10,dtype=float),'Total Replay':np.zeros(10,dtype=float),
          'Partial Replay':np.zeros(10,dtype=float),
          'SynF (constrained)':np.zeros(10,dtype=float), 'LwF':np.zeros(10,dtype=float),
          'A-GEM':np.zeros(10,dtype=float), 'None':np.zeros(10,dtype=float)}


for count,name in enumerate(bte_end.keys()):
    #print(name, count)
    for i in range(10):
        if count <8:
            bte_end[name][i] = np.log(btes_top[count][i][9-i])
        else:
            bte_end[name][i] = np.log(btes_bottom[count-8][i][9-i])

tmp_ble = {}
for id in combined_alg_name:
    tmp_ble[id] = bte_end[id]

df_ble = pd.DataFrame.from_dict(tmp_ble)
df_ble = pd.melt(df_ble,var_name='Algorithms', value_name='Backward Transfer Efficieny')

fte_end = {'SynN*':np.zeros(10,dtype=float), 'SynF*':np.zeros(10,dtype=float), 
          'Model Zoo*':np.zeros(10,dtype=float),
          'ProgNN*':np.zeros(10,dtype=float), 'LMC*':np.zeros(10,dtype=float),
          'DF-CNN*':np.zeros(10,dtype=float),'Total Replay':np.zeros(10,dtype=float),
          'Partial Replay':np.zeros(10,dtype=float),
          'SynF (constrained)':np.zeros(10,dtype=float), 'LwF':np.zeros(10,dtype=float),
          'A-GEM':np.zeros(10,dtype=float), 'None':np.zeros(10,dtype=float)}


for count,name in enumerate(fte_end.keys()):
    #print(name, count)
    for i in range(10):
        if count <8:
            fte_end[name][i] = np.log(ftes_top[count][i])
        else:
            fte_end[name][i] = np.log(ftes_bottom[count-8][i])

tmp_fle = {}
for id in combined_alg_name:
    tmp_fle[id] = fte_end[id]

df_fle = pd.DataFrame.from_dict(tmp_fle)
df_fle = pd.melt(df_fle,var_name='Algorithms', value_name='Forward Transfer Efficieny')

#%%
btes_all['cifar'] = df_ble
ftes_all['cifar'] = df_fle
tes_all['cifar'] = df_le
labels.append(combined_alg_name)
# %%
### MAIN HYPERPARAMS ###
task_num = 6
total_alg = 7
combined_alg_name = ['SynN*','SynF*', 'Model Zoo*','LwF', 'Total Replay', 'Partial Replay', 'None']
btes = [[] for i in range(total_alg)]
ftes = [[] for i in range(total_alg)]
tes = [[] for i in range(total_alg)]
model_file_combined = ['odin','odif', 'model_zoo', 'LwF', 'offline', 'exact', 'None']
########################

#%% code for 500 samples
reps = 10

for alg in range(total_alg): 
    count = 0 
    bte_tmp = [[] for _ in range(reps)]
    fte_tmp = [[] for _ in range(reps)] 
    te_tmp = [[] for _ in range(reps)]

    for rep in range(reps):
        filename = 'spoken_digit/'+model_file_combined[alg]+'-'+str(rep)+'.pickle'

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
    
    print('Algo name:' , combined_alg_name[alg])
    print('Accuracy', np.round(calc_acc(err, task_num, reps),2))
    print('forget', np.round(calc_forget(err, task_num, reps),2))
    print('transfer', np.round(calc_transfer(err, single_err, task_num, reps),2))

#%%
te = {'SynN*':np.zeros(6,dtype=float), 'SynF*':np.zeros(6,dtype=float), 
    'Model Zoo*':np.zeros(6,dtype=float), 'LwF':np.zeros(6,dtype=float), 
    'Total Replay':np.zeros(6,dtype=float), 'Partial Replay':np.zeros(6,dtype=float), 
    'None':np.zeros(6,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(6):
        te[name][i] = np.log(tes[count][i][5-i])

mean_val = []
for name in te.keys():
    mean_val.append(np.mean(te[name]))
    print(name, np.round(np.mean(te[name]),2), np.round(np.std(te[name], ddof=1),2))

arg = np.argsort(mean_val)[::-1]
ordr.append(arg)
algos = list(te.keys())
combined_alg_name = []

for ii in arg:
    combined_alg_name.append(
        algos[ii]
    )
    
tmp_te = {}
for id in combined_alg_name:
    tmp_te[id] = te[id]

df_le = pd.DataFrame.from_dict(tmp_te)
df_le = pd.melt(df_le,var_name='Algorithms', value_name='Transfer Efficieny')

# %%
bte_end = {'SynN*':np.zeros(6,dtype=float), 'SynF*':np.zeros(6,dtype=float), 
    'Model Zoo*':np.zeros(6,dtype=float), 'LwF':np.zeros(6,dtype=float), 
    'Total Replay':np.zeros(6,dtype=float), 'Partial Replay':np.zeros(6,dtype=float), 
    'None':np.zeros(6,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(6):
        bte_end[name][i] = np.log(btes[count][i][5-i])

tmp_ble = {}
for id in combined_alg_name:
    tmp_ble[id] = bte_end[id]

df_ble = pd.DataFrame.from_dict(tmp_ble)
df_ble = pd.melt(df_ble,var_name='Algorithms', value_name='Backward Transfer Efficieny')

fte_end = {'SynN*':np.zeros(6,dtype=float), 'SynF*':np.zeros(6,dtype=float), 
    'Model Zoo*':np.zeros(6,dtype=float), 'LwF':np.zeros(6,dtype=float), 
    'Total Replay':np.zeros(6,dtype=float), 'Partial Replay':np.zeros(6,dtype=float), 
    'None':np.zeros(6,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(1,6):
        fte_end[name][i] = np.log(ftes[count][i])

tmp_fle = {}
for id in combined_alg_name:
    tmp_fle[id] = fte_end[id]

df_fle = pd.DataFrame.from_dict(fte_end)
df_fle = pd.melt(df_fle,var_name='Algorithms', value_name='Forward Transfer Efficieny')

#%%
btes_all['speech'] = df_ble
ftes_all['speech'] = df_fle
tes_all['speech'] = df_le
labels.append(combined_alg_name)

#%%

### MAIN HYPERPARAMS ###
task_num = 50
total_alg = 4
combined_alg_name = ['SynN*','SynF*', 'Model Zoo*', 'LwF']
btes = [[] for i in range(total_alg)]
ftes = [[] for i in range(total_alg)]
tes = [[] for i in range(total_alg)]
model_file_combined = ['synn','synf', 'model_zoo', 'LwF']
########################

#%% 

for alg in range(total_alg): 

    filename = 'food1k/results/'+model_file_combined[alg]+'.pickle'

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
    
    print('Algo name:' , combined_alg_name[alg])
    print('Accuracy', np.round(calc_acc(err, task_num, 1),2))
    print('forget', np.round(calc_forget(err, task_num, 1),2))
    print('transfer', np.round(calc_transfer(err, single_err, task_num, 1),2))

#%%
te = {'SynN*':np.zeros(50,dtype=float), 'SynF*':np.zeros(50,dtype=float), 'Model Zoo*':np.zeros(50,dtype=float), 
    'LwF':np.zeros(50,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(50):
        te[name][i] = np.log(tes[count][i][49-i])


mean_val = []
for name in te.keys():
    mean_val.append(np.mean(te[name]))
    print(name, np.round(np.mean(te[name]),2), np.round(np.std(te[name], ddof=1),2))

arg = np.argsort(mean_val)[::-1]
ordr.append(arg)
algos = list(te.keys())
combined_alg_name = []

for ii in arg:
    combined_alg_name.append(
        algos[ii]
    )
    
tmp_te = {}
for id in combined_alg_name:
    tmp_te[id] = te[id]

df_le = pd.DataFrame.from_dict(tmp_te)
df_le = pd.melt(df_le,var_name='Algorithms', value_name='Transfer Efficieny')



#%%
bte_end = {'SynN*':np.zeros(50,dtype=float), 'SynF*':np.zeros(50,dtype=float), 'Model Zoo*':np.zeros(50,dtype=float), 
    'LwF':np.zeros(50,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(50):
        bte_end[name][i] = np.log(btes[count][i][49-i])

tmp_ble = {}
for id in combined_alg_name:
    tmp_ble[id] = bte_end[id]

df_ble = pd.DataFrame.from_dict(tmp_ble)
df_ble = pd.melt(df_ble,var_name='Algorithms', value_name='Backward Transfer Efficieny')

fte_end = {'SynN*':np.zeros(50,dtype=float), 'SynF*':np.zeros(50,dtype=float), 'Model Zoo*':np.zeros(50,dtype=float), 
    'LwF':np.zeros(50,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(1,50):
        fte_end[name][i] = np.log(ftes[count][i])

tmp_fle = {}
for id in combined_alg_name:
    tmp_fle[id] = fte_end[id]

df_fle = pd.DataFrame.from_dict(fte_end)
df_fle = pd.melt(df_fle,var_name='Algorithms', value_name='Forward Transfer Efficieny')

#%%
btes_all['food1k'] = df_ble
ftes_all['food1k'] = df_fle
tes_all['food1k'] = df_le
labels.append(combined_alg_name)

#%%

### MAIN HYPERPARAMS ###
task_num = 20
total_alg = 8
combined_alg_name = ['SynN*','SynF*', 'Model Zoo*', 'LwF', 'A-GEM', 'Total Replay', 'Partial Replay', 'None']
btes = [[] for i in range(total_alg)]
ftes = [[] for i in range(total_alg)]
tes = [[] for i in range(total_alg)]
model_file_combined = ['synn','synf', 'model_zoo', 'LwF', 'agem', 'offline', 'exact', 'None']
########################

#%% 
reps = 1

for alg in range(total_alg): 
    count = 0 

    for rep in range(reps):
        filename = 'mini_imagenet/results/'+model_file_combined[alg]+'.pickle'

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
    
    print('Algo name:' , combined_alg_name[alg])
    print('Accuracy', np.round(calc_acc(err, task_num, reps),2))
    print('forget', np.round(calc_forget(err, task_num, reps),2))
    print('transfer', np.round(calc_transfer(err, single_err, task_num, reps),2))

#%%
te = {'SynN*':np.zeros(20,dtype=float), 'SynF*':np.zeros(20,dtype=float), 'Model Zoo*':np.zeros(20,dtype=float), 
    'LwF':np.zeros(20,dtype=float), 'A-GEM':np.zeros(20,dtype=float),
    'Total Replay':np.zeros(20,dtype=float), 'Partial Replay':np.zeros(20,dtype=float), 
    'None':np.zeros(20,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(20):
        te[name][i] = np.log(tes[count][i][19-i])


mean_val = []
for name in te.keys():
    mean_val.append(np.mean(te[name]))
    print(name, np.round(np.mean(te[name]),2), np.round(np.std(te[name], ddof=1),2))

arg = np.argsort(mean_val)[::-1]
ordr.append(arg)
algos = list(te.keys())
combined_alg_name = []

for ii in arg:
    combined_alg_name.append(
        algos[ii]
    )
    
tmp_te = {}
for id in combined_alg_name:
    tmp_te[id] = te[id]

df_le = pd.DataFrame.from_dict(tmp_te)
df_le = pd.melt(df_le,var_name='Algorithms', value_name='Transfer Efficieny')


#%%
fle['imagenet'] = np.mean(np.log(ftes), axis=1)
ble['imagenet'] = []
le['imagenet'] = []

bte_end = {'SynN*':np.zeros(20,dtype=float), 'SynF*':np.zeros(20,dtype=float), 'Model Zoo*':np.zeros(20,dtype=float), 
    'LwF':np.zeros(20,dtype=float), 'A-GEM':np.zeros(20,dtype=float),
    'Total Replay':np.zeros(20,dtype=float), 'Partial Replay':np.zeros(20,dtype=float), 
    'None':np.zeros(20,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(20):
        bte_end[name][i] = np.log(btes[count][i][19-i])

tmp_ble = {}
for id in combined_alg_name:
    tmp_ble[id] = bte_end[id]

df_ble = pd.DataFrame.from_dict(tmp_ble)
df_ble = pd.melt(df_ble,var_name='Algorithms', value_name='Backward Transfer Efficieny')

fte_end = {'SynN*':np.zeros(20,dtype=float), 'SynF*':np.zeros(20,dtype=float), 'Model Zoo*':np.zeros(20,dtype=float), 
    'LwF':np.zeros(20,dtype=float), 'A-GEM':np.zeros(20,dtype=float),
    'Total Replay':np.zeros(20,dtype=float), 'Partial Replay':np.zeros(20,dtype=float), 
    'None':np.zeros(20,dtype=float)}
for count,name in enumerate(te.keys()):
    for i in range(1,20):
        fte_end[name][i] = np.log(ftes[count][i])

tmp_fle = {}
for id in combined_alg_name:
    tmp_fle[id] = fte_end[id]

df_fle = pd.DataFrame.from_dict(fte_end)
df_fle = pd.melt(df_fle,var_name='Algorithms', value_name='Forward Transfer Efficieny')

#%%
btes_all['imagenet'] = df_ble
ftes_all['imagenet'] = df_fle
tes_all['imagenet'] = df_le
labels.append(combined_alg_name)


#%%
### MAIN HYPERPARAMS ###
task_num = 5
total_alg = 8
combined_alg_name = ['SynN*','SynF*', 'Model Zoo*', 'LwF', 'A-GEM', 'Total Replay', 'Partial Replay', 'None']
btes = [[] for i in range(total_alg)]
ftes = [[] for i in range(total_alg)]
tes = [[] for i in range(total_alg)]
model_file_combined = ['synn','synf', 'model_zoo', 'LwF', 'agem', 'offline', 'exact', 'None']
########################

#%% code for 500 samples
reps = 1

for alg in range(total_alg): 
    count = 0 

    for rep in range(reps):
        filename = 'five_datasets/results/'+model_file_combined[alg]+'.pickle'

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
    
    print('Algo name:' , combined_alg_name[alg])
    print('Accuracy', np.round(calc_acc(err, task_num, reps),2))
    print('forget', np.round(calc_forget(err, task_num, reps),2))
    print('transfer', np.round(calc_transfer(err, single_err, task_num, reps),2))

#%%
te = {'SynN*':np.zeros(5,dtype=float), 'SynF*':np.zeros(5,dtype=float), 'Model Zoo*':np.zeros(5,dtype=float), 
    'LwF':np.zeros(5,dtype=float), 'A-GEM':np.zeros(5,dtype=float),
    'Total Replay':np.zeros(5,dtype=float), 'Partial Replay':np.zeros(5,dtype=float), 
    'None':np.zeros(5,dtype=float)}

for count,name in enumerate(te.keys()):
    for i in range(5):
        te[name][i] = np.log(tes[count][i][4-i])

mean_val = []
for name in te.keys():
    mean_val.append(np.mean(te[name]))
    print(name, np.round(np.mean(te[name]),2), np.round(np.std(te[name], ddof=1),2))

arg = np.argsort(mean_val)[::-1]
ordr.append(arg)
algos = list(te.keys())
combined_alg_name = []

for ii in arg:
    combined_alg_name.append(
        algos[ii]
    )
    
tmp_te = {}
for id in combined_alg_name:
    tmp_te[id] = te[id]

df_le = pd.DataFrame.from_dict(tmp_te)
df_le = pd.melt(df_le,var_name='Algorithms', value_name='Transfer Efficieny')

#%%
bte_end = {'SynN*':np.zeros(5,dtype=float), 'SynF*':np.zeros(5,dtype=float), 'Model Zoo*':np.zeros(5,dtype=float), 
    'LwF':np.zeros(5,dtype=float), 'A-GEM':np.zeros(5,dtype=float),
    'Total Replay':np.zeros(5,dtype=float), 'Partial Replay':np.zeros(5,dtype=float), 
    'None':np.zeros(5,dtype=float)}
for count,name in enumerate(te.keys()):
    for i in range(5):
        bte_end[name][i] = np.log(btes[count][i][4-i])

tmp_ble = {}
for id in combined_alg_name:
    tmp_ble[id] = bte_end[id]

df_ble = pd.DataFrame.from_dict(tmp_ble)
df_ble = pd.melt(df_ble,var_name='Algorithms', value_name='Backward Transfer Efficieny')

fte_end = {'SynN*':np.zeros(5,dtype=float), 'SynF*':np.zeros(5,dtype=float), 'Model Zoo*':np.zeros(5,dtype=float), 
    'LwF':np.zeros(5,dtype=float), 'A-GEM':np.zeros(5,dtype=float),
    'Total Replay':np.zeros(5,dtype=float), 'Partial Replay':np.zeros(5,dtype=float), 
    'None':np.zeros(5,dtype=float)}
for count,name in enumerate(te.keys()):
    for i in range(1,5):
        fte_end[name][i] = np.log(ftes[count][i])

tmp_fle = {}
for id in combined_alg_name:
    tmp_fle[id] = fte_end[id]

df_fle = pd.DataFrame.from_dict(fte_end)
df_fle = pd.melt(df_fle,var_name='Algorithms', value_name='Forward Transfer Efficieny')

#%%
btes_all['five_dataset'] = df_ble
ftes_all['five_dataset'] = df_fle
tes_all['five_dataset'] = df_le
labels.append(combined_alg_name)


#%%
datasets = ['CIFAR 10X10', 'Speech', 'FOOD1k', 'Split Mini-Imagenet', '5-dataset']
FLE_yticks = [[-.3,0,.3], [-1.5,0,1], [-.1,0,.4], [-0.4,0,.6], [-1.5,0,.3]]
BLE_yticks = [[-.4,0,.2], [-3,0,2], [-.3,0,.3], [-0.6,0,.2], [-2.5,0,.5]]
LE_yticks = [[-.4,0,.2], [-3,0,2], [-.3,0,.4], [-0.6,0,.6], [-2.5,0,.4]]

xcolor = [["#984ea3","#984ea3","#984ea3","#984ea3","#4daf4a","#4daf4a","#984ea3","#984ea3","#4daf4a","#4daf4a","#4daf4a","#4daf4a","#4daf4a","#4daf4a","#4daf4a","#4daf4a","#4daf4a"],
          ["#984ea3", "#984ea3", "#984ea3", "#4daf4a", "#4daf4a", "#4daf4a", "#4daf4a", "#4daf4a", "#4daf4a", "#4daf4a"],
          ["#984ea3", "#984ea3", "#984ea3", "#4daf4a"],
          ["#984ea3", "#984ea3", "#4daf4a", "#4daf4a", "#984ea3", "#4daf4a", "#4daf4a", "#4daf4a", "#4daf4a", "#4daf4a", "#4daf4a", "#4daf4a", "#4daf4a"],
          ["#984ea3", "#984ea3", "#4daf4a", "#4daf4a", "#984ea3", "#4daf4a", "#4daf4a", "#4daf4a", "#4daf4a", "#4daf4a", "#4daf4a", "#4daf4a", "#4daf4a"]]

clr = [["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#984ea3", "#f781bf", "#b15928", "#b15928", "#e41a1c", "#f781bf", "#b15928", "#b15928"],
       ["#377eb8", "#e41a1c", "#4daf4a", "#f781bf", "#b15928", "#b15928", "#b15928"],
       ["#377eb8", "#e41a1c", "#4daf4a", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928", "#b15928"],
       ["#377eb8", "#e41a1c", "#4daf4a", "#f781bf", "#b15928", "#b15928", "#b15928", "#b15928"],
       ["#377eb8", "#e41a1c", "#4daf4a", "#f781bf", "#b15928", "#b15928", "#b15928", "#b15928"]]
fig, ax = plt.subplots(3,len(tes_all.keys()), figsize=(30,16))
sns.set_context('talk')

for ii, data in enumerate(tes_all.keys()):
    c_ = []
    for id in ordr[ii]:
        c_.append(clr[ii][id])

    clr_ = sns.color_palette(c_, n_colors=len(clr[ii]))
    ax_ = sns.stripplot(x='Algorithms', y='Forward Transfer Efficieny', data=ftes_all[data], hue='Algorithms', palette=clr_, ax=ax[0][ii], size=12, legend=None, alpha=.3)
    ax_.set_title(data, fontsize=38)
    ax_.set_xticklabels([])
    ax_.hlines(0, -1,len(labels[ii]), colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

    ax_.set_xlabel('')
    ax_.set_yticks(FLE_yticks[ii])
    ax_.tick_params('y',labelsize=30)
    if ii==0:
        ax_.set_ylabel('Forward Learning\n $\log$ FLE', fontsize=24)
    else:
        ax_.set_ylabel('$\log$ FLE', fontsize=24)

    right_side = ax_.spines["right"]
    right_side.set_visible(False)
    top_side = ax_.spines["top"]
    top_side.set_visible(False)

    ax_ = sns.stripplot(x='Algorithms', y='Backward Transfer Efficieny', data=btes_all[data], hue='Algorithms', palette=clr_, ax=ax[1][ii], size=12, legend=None, alpha=.3)

    ax_.set_xticklabels([])
    #ax_.set_xlim([0, len(labels[ii])])
    ax_.hlines(0, -1,len(labels[ii]), colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

    ax_.set_xlabel('')
    ax_.set_yticks(BLE_yticks[ii])
    ax_.tick_params('y', labelsize=30)
    if ii==0:
        ax_.set_ylabel('Backward Learning\n $\log$ BLE', fontsize=24)
    else:
        ax_.set_ylabel('$\log$ BLE', fontsize=24)

    right_side = ax_.spines["right"]
    right_side.set_visible(False)
    top_side = ax_.spines["top"]
    top_side.set_visible(False)

    ax_ = sns.stripplot(x='Algorithms', y='Transfer Efficieny', data=tes_all[data], hue='Algorithms', palette=clr_, ax=ax[2][ii], size=12, legend=None, alpha=.3)
    
    ax_.set_xticklabels(
    labels[ii],
    fontsize=18,rotation=65,ha="right",rotation_mode='anchor'
    )


    for xtick, color in zip(ax_.get_xticklabels(), c_):
        xtick.set_color(color)

    ax_.hlines(0, -1,len(labels[ii]), colors='grey', linestyles='dashed',linewidth=1.5, label='chance')

    ax_.set_xlabel('')
    ax_.set_yticks(LE_yticks[ii])
    ax_.tick_params('y', labelsize=30)
    if ii==0:
        ax_.set_ylabel('Overall Learning\n $\log$ LE', fontsize=24)
    else:
        ax_.set_ylabel('$\log$ LE', fontsize=24)

    right_side = ax_.spines["right"]
    right_side.set_visible(False)
    top_side = ax_.spines["top"]
    top_side.set_visible(False)

plt.savefig('stripplot_summary.pdf')
# %%
