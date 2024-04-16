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
# %%
alg_size = {
    'SiLLy-N-t' : [27.1, 53.2, 80.2, 107.12, 134.3, 161.1, 188.7, 217.1, 243.1, 272.11],
    'SiLLy-N-5' : [27.1, 53.2, 80.2, 107.12, 134.3, 134.3, 134.3, 134.3, 134.3, 134.3],
    'SiLLy-N-8' : [27.1, 53.2, 80.2, 107.12, 134.3, 161.1, 188.7, 217.1, 217.1, 217.1],
    'SiLLy-F-t' : [0.70, 2.38, 5.15, 9.03, 14.13, 20.20, 27.12, 35.30, 44.85, 55.42],
    'EWC' : [25.37, 76.07, 126.76, 177.46, 228.16, 278.86, 329.56, 380.26, 430.96, 481.66],
    'O-EWC' : [25.37, 76.07, 76.07, 76.07, 76.07, 76.07, 76.07, 76.07, 76.07, 76.07],
    'SI' : [50.72, 76.07, 76.07, 76.07, 76.07, 76.07, 76.07, 76.07, 76.07, 76.07],
    'LwF' : [25.37, 25.37, 25.37, 25.37, 25.37, 25.37, 25.37, 25.37, 25.37, 25.37], 
    'Model Zoo' : [27.6, 53.3, 80.2, 106.7, 134.0, 159.4, 186.8, 213.2, 238.2, 265.2],
    'ProgNN' : [25.8, 52, 78.3, 104.7, 131.1, 157.5, 184, 210.5, 236.2, 263.2],
    'DF-CNN' : [566.9, 566.9, 566.9, 566.9, 566.9, 566.9, 566.9, 566.9, 566.9, 566.9],
    'LMC' : [68.54, 70.56, 72.57, 74.58, 76.60, 78.61, 80.62, 82.63, 84.65, 86.66]
}

'''synn_size = {
    'encoder' : [26.4, 52.8, 79.2, 105.6, 132.0, 158.4, 184.8, 211.2, 237.6, 264.0],
    'channel' : [0.07, 0.28, 0.63, 1.12, 1.75, 2.52, 3.43, 4.48, 5.67, 7]
}'''
# %%
fig, ax = plt.subplots(1, 1, figsize=(9,8))
sns.set_context('talk')
tasks = np.arange(1,11,1)
clr = ['r', 'r', 'r', "#377eb8", "#b15928", "#f781bf", "#f781bf", "#f781bf", "#4daf4a", "#984ea3", "#f781bf", "#984ea3"]
marker_style = ['.', 'o', '*', '.', '.', 'o', '*', '.', '.', '.', '+', 'o',]
#"#984ea3""#984ea3""#4daf4a"
ax.plot(tasks, alg_size['SiLLy-N-t'], label='SiLLy-N-t', c=clr[0], marker=marker_style[0], markersize=12)
#ax.plot(tasks, synn_size['encoder'], label='SynN Encoder', c='k', marker=marker_style[0], markersize=12, linewidth=3)
#ax.plot(tasks, synn_size['channel'], label='SynN Channel', c='r', marker=marker_style[0], markersize=12)

for ii,key in enumerate(alg_size.keys()):
    if ii ==0:
        continue

    ax.plot(tasks, alg_size[key], label=key, c=clr[ii], marker=marker_style[ii], markersize=12, alpha=.8)


ax.set_xlim(1,10.5)
ax.set_yticks([0,300,600])
ax.set_ylabel('Model Size (MB)', fontsize=40)
ax.set_xlabel('Tasks Seen', fontsize=40)

ax.tick_params(labelsize=25)
fig.legend(bbox_to_anchor=(1.3, .95), fontsize=25, frameon=False)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

plt.savefig('model_size.pdf', bbox_inches='tight')
# %%
