import numpy as np
import pickle
import pandas as pd

n_shifts = 6

backward_dfs = []

for shift in range(n_shifts):
    df_of_split = pickle.load(open('BTE_shift_{}.p'.format(shift), 'rb'), encoding = 'latin1')
    backward_dfs.append(df_of_split)
    
backward_df = pd.concat(backward_dfs)

forward_dfs = []

for shift in range(n_shifts):
    df_of_split = pickle.load(open('FTE_shift_{}.p'.format(shift), 'rb'), encoding = 'latin1')
    forward_dfs.append(df_of_split)
    
forward_df = pd.concat(forward_dfs)

pickle.dump((forward_df, backward_df), open('FTE_BTE_tuple.p', 'wb'))
