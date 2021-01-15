#%%
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
# %%
alg_name = ['PLN','PLF','Prog_NN', 'DF_CNN','LwF','EWC','O-EWC','SI', 'Replay \n (increasing amount)', 'Replay \n (fixed amount)', 'None']
model_file = ['dnn','uf','Prog_NN','DF_CNN', 'LwF', 'EWC', 'OEWC', 'SI', 'offline', 'exact', 'None']
total_alg = 11
slots = 10
shifts = 6
time_info = [[] for i in range(total_alg)]
mem_info = [[] for i in range(total_alg)]

for alg in range(total_alg): 
    if alg < 2:
        filename = './result/time_res/'+model_file[alg]+'_same_machine.pickle'
    elif alg == 2 or alg == 3:
        filename = './result/time_res/'+model_file[alg]+str(1)+'_'+str(1)+'.pickle'
    else:
        filename = './result/time_res/'+model_file[alg]+'-'+str(1)+'-'+str(1)+'.pkl'

    with open(filename,'rb') as f:
            data = pickle.load(f)

    time_info[alg].extend(np.asarray(data))
# %%
for alg in range(total_alg): 
    if alg < 2:
        filename = './result/mem_res/'+model_file[alg]+'_same_machine.pickle'
    elif alg == 2 or alg == 3:
        filename = './result/mem_res/'+model_file[alg]+str(1)+'_'+str(1)+'.pickle'
    else:
        filename = './result/mem_res/'+model_file[alg]+'-'+str(1)+'-'+str(1)+'.pkl'

    with open(filename,'rb') as f:
            data = pickle.load(f)

    mem_info[alg].extend(np.asarray(data)/1024)
# %%
fontsize=20
ticksize=18
fig = plt.figure(constrained_layout=True,figsize=(14,5))
#fig, ax = plt.subplots(1,2, figsize=(14,5))
gs = fig.add_gridspec(5, 14)

clr = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#f781bf", "#f781bf", "#f781bf", "#f781bf", "#b15928", "#b15928", "#b15928"]
c = sns.color_palette(clr, n_colors=len(clr))
marker_style = ['.', '.', '.', '.', '.', '+', 'o', '*', '.', '+', 'o']
task_sample = [5000, 9500, 13500, 17000, 20000, 22500, 24500, 26000, 27000, 27500]

ax = fig.add_subplot(gs[:5,:5])
for alg_no,alg in enumerate(alg_name):
    if alg_no<2:
        ax.plot(task_sample,time_info[alg_no], c=c[alg_no], label=alg_name[alg_no], linewidth=3, marker=marker_style[alg_no])
    else:
        ax.plot(task_sample,time_info[alg_no], c=c[alg_no], label=alg_name[alg_no], marker=marker_style[alg_no])

#ax.set_yticks([1,10,20,30,40])
#ax.set_ylim([.9,45])
ax.set_yscale('log')
ax.set_xticks([5000, 13500, 20000, 27000])
ax.tick_params(labelsize=ticksize)
ax.set_xlabel('Number of training samples', fontsize=fontsize)
ax.set_ylabel('Time (s)', fontsize=fontsize)
#ax[0].set_title("Label Shuffled CIFAR", fontsize = fontsize)
#ax.hlines(1,1,10, colors='grey', linestyles='dashed',linewidth=1.5)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
plt.tight_layout()

ax = fig.add_subplot(gs[:5,6:11])
for alg_no,alg in enumerate(alg_name):
    if alg_no<2:
        ax.plot(task_sample,mem_info[alg_no], c=c[alg_no], label=alg_name[alg_no], linewidth=3, marker=marker_style[alg_no])
    else:
        ax.plot(task_sample,mem_info[alg_no], c=c[alg_no], label=alg_name[alg_no], marker=marker_style[alg_no])

#ax.set_yticks([1,6,12,16])
#ax.set_ylim([.9,16])
#ax.set_xticks([5000, 9500, 13500, 17000, 20000, 22500, 24500, 26000, 27000, 27500])
ax.set_yscale('log')
ax.set_xticks([5000, 13500, 20000, 27000])
ax.tick_params(labelsize=ticksize)
ax.set_xlabel('Number of training samples', fontsize=fontsize)
ax.set_ylabel('Memory (kB)', fontsize=fontsize)
#ax[1].set_title("Label Shuffled CIFAR", fontsize = fontsize)
#ax.hlines(1,1,10, colors='grey', linestyles='dashed',linewidth=1.5)
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)
handles, labels_ = ax.get_legend_handles_labels()
ax.legend(handles, labels_, bbox_to_anchor=(1, 1), fontsize=14, frameon=False)
plt.tight_layout()

plt.savefig('./result/figs/scaling_unnormalized.pdf')
# %% fit the time and memory curves
from scipy.optimize import curve_fit 
from math import fmod

def test_cons(x, b): 
    return b

def test_lin(x, a, b): 
    return (a*x) + b

def test_sqr(x, a, b): 
    return (a*x*x) + b

def test_cube(x, a, b): 
    return (a*x*x*x) + b

def test_quad(x, a, b): 
    return (a*x*x*x*x) + b

sample_no = np.asarray(
    [5000, 9500, 13500, 17000, 20000, 22500, 24500, 26000, 27000, 27500]
    )
fit_func = [2,1,4,2,2,2,2,2,3,3,2]
fig, ax = plt.subplots(3,4, figsize=(24,20))
sns.set_context("talk")

for i, times in enumerate(time_info):
    if fit_func[i] == 1:
        param, param_cov = curve_fit(test_lin, sample_no/1000, times)
        times_hat = test_lin(sample_no/1000, param[0], param[1])
        fit = 'linear'
    elif fit_func[i] == 2:
        param, param_cov = curve_fit(test_sqr, sample_no/1000, times)
        times_hat = test_sqr(sample_no/1000, param[0], param[1])
        fit = 'square'
    elif fit_func[i] == 3:
        param, param_cov = curve_fit(test_cube, sample_no/1000, times)
        times_hat = test_cube(sample_no/1000, param[0], param[1])
        fit = 'cube'
    elif fit_func[i] == 4:
        param, param_cov = curve_fit(test_quad, sample_no/1000, times)
        times_hat = test_quad(sample_no/1000, param[0], param[1])
        fit = 'quadruple'

    col, row = i//3, int(fmod(i,3))
    #print(row, col, i)
    ax[row][col].scatter(sample_no, times, c='r', label = 'true val')
    ax[row][col].plot(sample_no, times_hat, label = 'fitted val')
    ax[row][col].legend()
    ax[row][col].set_title(alg_name[i] + ' ' + fit+ ' a=' + str(np.round(param[0],2)) + ' b=' + str(np.round(param[1],2)))

plt.savefig('./result/figs/time_fitting.pdf')
# %%
fit_func = [1,1,3,1,2,2,2,2,2,2,2]
fig, ax = plt.subplots(3,4, figsize=(28,20))
sns.set_context("talk")

for i, mem in enumerate(mem_info):
    if fit_func[i] == 0:
        param, param_cov = curve_fit(test_cons, sample_no/1000, mem)
        mem_hat = test_cons(sample_no/1000, param)
        fit = 'constant'
    if fit_func[i] == 1:
        param, param_cov = curve_fit(test_lin, sample_no/1000, mem)
        mem_hat = test_lin(sample_no/1000, param[0], param[1])
        fit = 'linear'
    elif fit_func[i] == 2:
        param, param_cov = curve_fit(test_sqr, sample_no/1000, mem)
        mem_hat = test_sqr(sample_no/1000, param[0], param[1])
        fit = 'square'
    elif fit_func[i] == 3:
        param, param_cov = curve_fit(test_cube, sample_no/1000, mem)
        mem_hat = test_cube(sample_no/1000, param[0], param[1])
        fit = 'cube'

    col, row = i//3, int(fmod(i,3))
    print(row, col, i)
    ax[row][col].scatter(sample_no, mem, c='r', label = 'true val')
    ax[row][col].plot(sample_no, mem_hat, label = 'fitted val')
    ax[row][col].legend()
    ax[row][col].set_title(alg_name[i] + ' ' + fit+ ' a=' + str(np.round(param[0],2)) + ' b=' + str(np.round(param[1],2)))

    plt.savefig('./result/figs/mem_fitting.pdf')
# %% change fitting function according to jovo
from scipy.optimize import curve_fit 
from math import fmod
from numpy import log 

def test(x, a0, a1, a2, a3, a4, a_log): 
    return (a0 + a1*x + a2*x*x + a3*x*x*x + a4*x*x*x*x + a_log*x*log(x))

sample_no = np.asarray(
    [5000, 9500, 13500, 17000, 20000, 22500, 24500, 26000, 27000, 27500]
    )
sample_no_normalized = sample_no/1e4

fit_func = [2,1,4,2,2,2,2,2,3,3,2]
fig, ax = plt.subplots(3,4, figsize=(24,20))
sns.set_context("talk")

for i, times in enumerate(time_info):
    param, param_cov = curve_fit(test, sample_no_normalized, times)
    times_hat = test(sample_no_normalized, param[0], param[1], param[2], param[3], param[4], param[5])

    col, row = i//3, int(fmod(i,3))
    #print(row, col, i)
    ax[row][col].scatter(sample_no, times, c='r', label = 'true val')
    ax[row][col].plot(sample_no, times_hat, label = 'fitted val')
    ax[row][col].legend()
    #ax[row][col].set_title(alg_name[i] + ' ' + fit+ ' a=' + str(np.round(param[0],2)) + ' b=' + str(np.round(param[1],2)))

plt.savefig('./result/figs/time_fitting_jovo.pdf')
# %% change fitting function according to jovo
from scipy.optimize import curve_fit 
from math import fmod
from numpy import log 

def test1(x, a0, a1): 
    return a0 + a1*x

def test2(X, a0, a1, a2): 
    x, T = X
    return a0 + a1*x + a2*x*T

sample_no = np.asarray(
    [5000, 9500, 13500, 17000, 20000, 22500, 24500, 26000, 27000, 27500]
    )
sample_no_normalized = sample_no/1e4
T = list(range(1,11))

fig, ax = plt.subplots(3,4, figsize=(24,20))
sns.set_context("talk")

for i, times in enumerate(time_info):
    param1, param_cov1 = curve_fit(test1, sample_no_normalized, times)
    times_hat1 = test1(sample_no_normalized, param1[0], param1[1])

    param2, param_cov2 = curve_fit(test2, (sample_no_normalized, T), times)
    times_hat2 = test2((sample_no_normalized, T), param2[0], param2[1], param2[2])

    col, row = i//3, int(fmod(i,3))
    a00, a01 = np.round(param1[0],2), np.round(param1[1],2)
    a10, a11, a12 = np.round(param2[0],2), np.round(param2[1],2), np.round(param2[2],2)
    label1 = str(a00) + ' + ' + str(a01) + '*x'
    label2 = str(a10) + ' + ' + str(a11) + '*x' + ' + ' + str(a12) + '*x*T'
    #print(row, col, i)
    ax[row][col].scatter(sample_no, times, c='r', label = 'true val')
    ax[row][col].plot(sample_no, times_hat1, label = label1)
    ax[row][col].plot(sample_no, times_hat2, label = label2)
    ax[row][col].set_title(alg_name[i])
    ax[row][col].legend()
    #ax[row][col].set_title(alg_name[i] + ' ' + fit+ ' a=' + str(np.round(param[0],2)) + ' b=' + str(np.round(param[1],2)))

plt.savefig('./result/figs/time_fitting_jovo.pdf')

# %% change fitting function according to jovo
from scipy.optimize import curve_fit 
from math import fmod
from numpy import log 

def test1(x, a0, a1): 
    return a0 + a1*x

def test2(x, a0, a1): 
    return a0 + a1*x*log(x)

def test3(X, a0, a1, a2): 
    x, T = X
    return a0 + a1*x + a2*T*T

sample_no = np.asarray(
    [5000, 9500, 13500, 17000, 20000, 22500, 24500, 26000, 27000, 27500]
    )
sample_no_normalized = sample_no/1e4
T = list(range(1,11))

fig, ax = plt.subplots(3,4, figsize=(24,20))
sns.set_context("talk")

for i, mem in enumerate(mem_info):
    param1, param_cov1 = curve_fit(test1, sample_no_normalized, mem)
    times_hat1 = test1(sample_no_normalized, param1[0], param1[1])

    param2, param_cov2 = curve_fit(test2, (sample_no_normalized, T), mem)
    times_hat2 = test2((sample_no_normalized, T), param2[0], param2[1])

    param3, param_cov3 = curve_fit(test3, (sample_no_normalized, T), mem)
    times_hat3 = test3((sample_no_normalized, T), param3[0], param3[1], param3[2])

    col, row = i//3, int(fmod(i,3))
    a00, a01 = np.round(param1[0],2), np.round(param1[1],2)
    a10, a11 = np.round(param2[0],2), np.round(param2[1],2)
    a20, a21, a22 = np.round(param3[0],2), np.round(param3[1],2), np.round(param3[2],2)

    label1 = str(a00) + ' + ' + str(a01) + '*x'
    label2 = str(a10) + ' + ' + str(a11) + '*x' + ' + ' + str(a12) + '*x*T'
    #print(row, col, i)
    ax[row][col].scatter(sample_no, times, c='r', label = 'true val')
    ax[row][col].plot(sample_no, times_hat1, label = label1)
    ax[row][col].plot(sample_no, times_hat2, label = label2)
    ax[row][col].legend()
    #ax[row][col].set_title(alg_name[i] + ' ' + fit+ ' a=' + str(np.round(param[0],2)) + ' b=' + str(np.round(param[1],2)))

plt.savefig('./result/figs/mem_fitting_jovo.pdf')

# %%
