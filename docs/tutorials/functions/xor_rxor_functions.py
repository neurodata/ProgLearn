import numpy as np
import random

from math import log2, ceil

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from proglearn.forest import LifelongClassificationForest, UncertaintyForest
from proglearn.sims import *
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.transformers import TreeClassificationTransformer, NeuralClassificationTransformer
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter

from joblib import Parallel, delayed

def get_colors(colors, inds):
    c = [colors[i] for i in inds]
    return c


def plot_xor_rxor(data, labels, title):
    colors = sns.color_palette("Dark2", n_colors=2)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(data[:, 0], data[:, 1], c=get_colors(colors, labels), s=50)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=30)
    #plt.tight_layout()
    ax.axis("off")
    plt.show()

def run(mc_rep, n_test, n_trees,n_xor,n_rxor, mean_error, std_error, mean_te, std_te):
    for i,n1 in enumerate(n_xor):
        #print('starting to compute %s xor\n'%n1)
        error = np.array(
            Parallel(n_jobs=-1,verbose=0)(
            delayed(experiment)(
                n1,0,task2_angle=np.pi/4,
                max_depth=ceil(log2(n1))
            ) for _ in range(mc_rep)
        )
        )
        mean_error[:,i] = np.mean(error,axis=0)
        std_error[:,i] = np.std(error,ddof=1,axis=0)
        mean_te[0,i] = np.mean(error[:,0])/np.mean(error[:,1])
        mean_te[1,i] = np.mean(error[:,2])/np.mean(error[:,3])
        mean_te[2,i] = np.mean(error[:,0])/np.mean(error[:,4])
        mean_te[3,i] = np.mean(error[:,2])/np.mean(error[:,5])

        if n1==n_xor[-1]:
            for j,n2 in enumerate(n_rxor):
                #print('starting to compute %s rxor\n'%n2)
                
                error = np.array(
                    Parallel(n_jobs=-1,verbose=0)(
                    delayed(experiment)(
                        n1,n2,task2_angle=np.pi/4,
                        max_depth=ceil(log2(750))
                    ) for _ in range(mc_rep)
                )
                )
                mean_error[:,i+j+1] = np.mean(error,axis=0)
                std_error[:,i+j+1] = np.std(error,ddof=1,axis=0)
                mean_te[0,i+j+1] = np.mean(error[:,0])/np.mean(error[:,1])
                mean_te[1,i+j+1] = np.mean(error[:,2])/np.mean(error[:,3])
                mean_te[2,i+j+1] = np.mean(error[:,0])/np.mean(error[:,4])
                mean_te[3,i+j+1] = np.mean(error[:,2])/np.mean(error[:,5])

    return mean_error, std_error, mean_te, std_te


def experiment(n_task1, n_task2, n_test=1000, 
               task1_angle=0, task2_angle=np.pi/2, 
               n_trees=10, max_depth=None, random_state=None):
    
    """
    A function to do progressive experiment between two tasks
    where the task data is generated using Gaussian parity.
    
    Parameters
    ----------
    n_task1 : int
        Total number of train sample for task 1.
    
    n_task2 : int
        Total number of train dsample for task 2

    n_test : int, optional (default=1000)
        Number of test sample for each task.
        
    task1_angle : float, optional (default=0)
        Angle in radian for task 1.
            
    task2_angle : float, optional (default=numpy.pi/2)
        Angle in radian for task 2.
            
    n_trees : int, optional (default=10)
        Number of total trees to train for each task.

    max_depth : int, optional (default=None)
        Maximum allowable depth for each tree.
        
    random_state : int, RandomState instance, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        
    
    Returns
    -------
    errors : array of shape [6]
        Elements of the array is organized as single task error task1,
        multitask error task1, single task error task2,
        multitask error task2, naive UF error task1,
        naive UF task2.
    """

    if n_task1==0 and n_task2==0:
        raise ValueError('Wake up and provide samples to train!!!')

    if random_state != None:
        np.random.seed(random_state)

    errors = np.zeros(6,dtype=float)

    default_transformer_class = TreeClassificationTransformer
    default_transformer_kwargs = {"kwargs" : {"max_depth" : max_depth}}

    default_voter_class = TreeClassificationVoter
    default_voter_kwargs = {}

    default_decider_class = SimpleArgmaxAverage
    default_decider_kwargs = {"classes" : np.arange(2)}
    progressive_learner = ProgressiveLearner(default_transformer_class = default_transformer_class,
                                            default_transformer_kwargs = default_transformer_kwargs,
                                            default_voter_class = default_voter_class,
                                            default_voter_kwargs = default_voter_kwargs,
                                            default_decider_class = default_decider_class,
                                            default_decider_kwargs = default_decider_kwargs)
    uf = ProgressiveLearner(default_transformer_class = default_transformer_class,
                                            default_transformer_kwargs = default_transformer_kwargs,
                                            default_voter_class = default_voter_class,
                                            default_voter_kwargs = default_voter_kwargs,
                                            default_decider_class = default_decider_class,
                                            default_decider_kwargs = default_decider_kwargs)
    naive_uf = ProgressiveLearner(default_transformer_class = default_transformer_class,
                                            default_transformer_kwargs = default_transformer_kwargs,
                                            default_voter_class = default_voter_class,
                                            default_voter_kwargs = default_voter_kwargs,
                                            default_decider_class = default_decider_class,
                                            default_decider_kwargs = default_decider_kwargs)
    
    #source data
    X_task1, y_task1 = generate_gaussian_parity(n_task1, angle_params=task1_angle)
    test_task1, test_label_task1 = generate_gaussian_parity(n_test, angle_params=task1_angle)

    #target data
    X_task2, y_task2 = generate_gaussian_parity(n_task2, angle_params=task2_angle)
    test_task2, test_label_task2 = generate_gaussian_parity(n_test, angle_params=task2_angle)

    if n_task1 == 0:
        progressive_learner.add_task(X_task2, y_task2, num_transformers=n_trees)

        errors[0] = 0.5
        errors[1] = 0.5

        uf_task2=progressive_learner.predict(test_task2,
                                             transformer_ids=[0], task_id=0)
        l2f_task2=progressive_learner.predict(test_task2, task_id=0)

        errors[2] = 1 - np.mean(uf_task2 == test_label_task2)
        errors[3] = 1 - np.mean(l2f_task2 == test_label_task2)
        
        errors[4] = 0.5
        errors[5] = 1 - np.mean(uf_task2 == test_label_task2)
    elif n_task2 == 0:
        progressive_learner.add_task(X_task1, y_task1,
                                     num_transformers=n_trees)

        uf_task1=progressive_learner.predict(test_task1, 
                                             transformer_ids=[0], task_id=0)
        l2f_task1=progressive_learner.predict(test_task1, task_id=0)

        errors[0] = 1 - np.mean(uf_task1 == test_label_task1)
        errors[1] = 1 - np.mean(l2f_task1 == test_label_task1)
        
        errors[2] = 0.5
        errors[3] = 0.5
        
        errors[4] = 1 - np.mean(uf_task1 == test_label_task1)
        errors[5] = 0.5
    else:
        progressive_learner.add_task(X_task1, y_task1, num_transformers=n_trees)
        progressive_learner.add_task(X_task2, y_task2, num_transformers=n_trees)

        uf.add_task(X_task1, y_task1, num_transformers=2*n_trees)
        uf.add_task(X_task2, y_task2, num_transformers=2*n_trees)
        
        naive_uf_train_x = np.concatenate((X_task1,X_task2),axis=0)
        naive_uf_train_y = np.concatenate((y_task1,y_task2),axis=0)
        naive_uf.add_task(
                naive_uf_train_x, naive_uf_train_y, num_transformers=n_trees
                )
        
        uf_task1=uf.predict(test_task1, transformer_ids=[0], task_id=0)
        l2f_task1=progressive_learner.predict(test_task1, task_id=0)
        uf_task2=uf.predict(test_task2, transformer_ids=[1], task_id=1)
        l2f_task2=progressive_learner.predict(test_task2, task_id=1)
        naive_uf_task1 = naive_uf.predict(
            test_task1, transformer_ids=[0], task_id=0
        )
        naive_uf_task2 = naive_uf.predict(
            test_task2, transformer_ids=[0], task_id=0
        )

        errors[0] = 1 - np.mean(
            uf_task1 == test_label_task1
        )
        errors[1] = 1 - np.mean(
            l2f_task1 == test_label_task1
        )
        errors[2] = 1 - np.mean(
            uf_task2 == test_label_task2
        )
        errors[3] = 1 - np.mean(
            l2f_task2 == test_label_task2
        )
        errors[4] = 1 - np.mean(
            naive_uf_task1 == test_label_task1
        )
        errors[5] = 1 - np.mean(
            naive_uf_task2 == test_label_task2
        )

    return errors

def plot_error_and_eff(n1s, n2s, mean_error, mean_te, TASK1, TASK2):
    """
    A function that plots the generalization error and
    transfer efficiency for the xor-nxor-rxor experiment
    
    Parameters
    ----------
    n1s : range(int)
        Array of sample sizes tested for the first learner.

    n2s : range(int)
        Array of sample sizes tested for the second learner.

    mean_error : np.array
        Array of generalization errors.

    mean_te : np.array
        Array of transfer efficiencies.
    
    task1 : str
        String of the name of the first task.

    task2 : str
        String of the name of the second task.
    """

    ns = np.concatenate((n1s, n2s + n1s[-1]))
    ls=['-', '--']

    ################################
    # Plots of Generalization Error
    ################################
    algorithms = ["XOR Forest", "N-XOR Forest", "PLF", "RF"]


    fontsize=30
    labelsize=28

    colors = sns.color_palette("Set1", n_colors = 2)

    fig = plt.figure(constrained_layout=True,figsize=(21,14))
    gs = fig.add_gridspec(14, 21)
    ax1 = fig.add_subplot(gs[7:,:6])
    ax1.plot(ns, mean_error[1], label=algorithms[2], c=colors[0], ls=ls[np.sum(1 > 1).astype(int)], lw=3)
    ax1.plot(ns, mean_error[4], label=algorithms[3], c='g', ls=ls[np.sum(1 > 1).astype(int)], lw=3)

    ax1.set_ylabel('Generalization Error (%s)'%(TASK1), fontsize=fontsize)
    ax1.legend(loc='upper left', fontsize=20, frameon=False)
    #ax1.set_ylim(0.09, 0.21)
    ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
    ax1.tick_params(labelsize=labelsize)
    #ax1.set_yticks([0.1,0.15, 0.2])
    ax1.set_xticks([50,750,1500])
    ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
    ax1.set_title('XOR', fontsize=30)

    right_side = ax1.spines["right"]
    right_side.set_visible(False)
    top_side = ax1.spines["top"]
    top_side.set_visible(False)

    ax1.text(400, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=26)
    ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=26)


    algorithms = ["XOR Forest", "N-XOR Forest", "PLF", "RF"]

    TASK1='XOR'
    TASK2='R-XOR'

    ax1 = fig.add_subplot(gs[7:,7:13])

    ax1.plot(ns[len(n1s):], mean_error[3, len(n1s):], label=algorithms[2], c=colors[0], lw=3)
    ax1.plot(ns[len(n1s):], mean_error[5, len(n1s):], label=algorithms[3], c='g', lw=3)

    ax1.set_ylabel('Generalization Error (%s)'%(TASK2), fontsize=fontsize)
    ax1.legend(loc='upper left', fontsize=20, frameon=False)
    ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
    ax1.tick_params(labelsize=labelsize)
    ax1.set_xticks([50,750,1500])
    ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")


    right_side = ax1.spines["right"]
    right_side.set_visible(False)
    top_side = ax1.spines["top"]
    top_side.set_visible(False)

    ax1.text(400, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=26)
    ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=26)

    ax1.set_title('R-XOR', fontsize=30)

    ################################
    # Plots of Transfer Efficiency
    ################################

    algorithms = ['PLF BTE', 'PLF FTE', 'RF BTE', 'RF FTE']

    ax1 = fig.add_subplot(gs[7:,14:])

    ax1.plot(ns, mean_te[0], label=algorithms[0], c=colors[0], ls=ls[0], lw=3)
    ax1.plot(ns[len(n1s):], mean_te[1, len(n1s):], label=algorithms[1], c=colors[0], ls=ls[1], lw=3)
    ax1.plot(ns, mean_te[2], label=algorithms[2], c='g', ls=ls[0], lw=3)
    ax1.plot(ns[len(n1s):], mean_te[3, len(n1s):], label=algorithms[3], c='g', ls=ls[1], lw=3)

    ax1.set_ylabel('Forward/Backward \n Transfer Efficiency (FTE/BTE)', fontsize=fontsize)
    ax1.legend(loc='upper left', fontsize=20, frameon=False)
    ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
    ax1.tick_params(labelsize=labelsize)
    #ax1.set_yticks([0,.5,1,1.5])
    ax1.set_xticks([50,750,1500])
    ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
    right_side = ax1.spines["right"]
    right_side.set_visible(False)
    top_side = ax1.spines["top"]
    top_side.set_visible(False)
    ax1.hlines(1, 50,1500, colors='gray', linestyles='dashed',linewidth=1.5)

    ax1.text(400, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=26)
    ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=26)

    colors = sns.color_palette('Dark2', n_colors=2)

    X, Y = generate_gaussian_parity(750, angle_params=0)
    Z, W = generate_gaussian_parity(750, angle_params=np.pi/4)

    ax = fig.add_subplot(gs[:6,4:10])
    clr = [colors[i] for i in Y]
    ax.scatter(X[:, 0], X[:, 1], c=clr, s=50)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Gaussian XOR', fontsize=30)

    #plt.tight_layout()
    ax.axis('off')

    colors = sns.color_palette('Dark2', n_colors=2)

    ax = fig.add_subplot(gs[:6,11:16])
    clr = [colors[i] for i in W]
    ax.scatter(Z[:, 0], Z[:, 1], c=clr, s=50)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Gaussian R-XOR', fontsize=30)
    ax.axis('off')

def bte_v_angle(angle_sweep,task1_sample,task2_sample,mc_rep):
    mean_te = np.zeros(len(angle_sweep), dtype=float)
    for ii,angle in enumerate(angle_sweep):
        error = np.array(
            Parallel(n_jobs=-1,verbose=0)(
            delayed(experiment)(
                task1_sample,task2_sample,
                task2_angle=angle*np.pi/180, 
                max_depth=ceil(log2(task1_sample))
            ) for _ in range(mc_rep)
        )
        )

        mean_te[ii] = np.mean(error[:,0])/np.mean(error[:,1])
    return mean_te

def plot_bte_v_angle(mean_te):
    angle_sweep = range(0,90,1)

    sns.set_context("talk")
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.plot(angle_sweep,mean_te,linewidth = 3)
    ax.set_xticks(range(0,91,10))
    ax.set_xlabel('Angle of Rotation (Degrees)')
    ax.set_ylabel('Backward Transfer Efficiency (XOR)')
    ax.hlines(1, 0,90, colors='gray', linestyles='dashed',linewidth=1.5)

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)

def bte_v_nsamples(task2_sample_sweep, task1_sample,task2_angle,mc_rep):
    mean_te = np.zeros(len(task2_sample_sweep), dtype=float)
    for ii,sample_no in enumerate(task2_sample_sweep):
        error = np.array(
            Parallel(n_jobs=-1,verbose=0)(
            delayed(experiment)(
                task1_sample,sample_no,
                task2_angle=task2_angle, 
                max_depth=None
            ) for _ in range(mc_rep)
        )
        )

        mean_te[ii] = np.mean(error[:,0])/np.mean(error[:,1])
    return mean_te

def plot_bte_v_nsamples(mean_te):
    task2_sample_sweep = (2**np.arange(np.log2(60), np.log2(5010)+1, .25)).astype('int')

    sns.set_context("talk")
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.plot(task2_sample_sweep,mean_te,linewidth = 3)
    ax.hlines(1, 60,5200, colors='gray', linestyles='dashed',linewidth=1.5)
    ax.set_xscale('log')
    ax.set_xticks([])
    #ax.set_yticks([0.87,0.9,0.93])
    ax.tick_params(labelsize=26)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.text(50, np.mean(ax.get_ylim())-.060, "50", fontsize=26)
    ax.text(500, np.mean(ax.get_ylim())-.060, "500", fontsize=26)
    ax.text(5000, np.mean(ax.get_ylim())-.060, "5000", fontsize=26)

    ax.text(50, np.mean(ax.get_ylim())-.068, "Number of $25^\circ$-RXOR Training Samples", fontsize=24)
    ax.set_ylabel('Backward Transfer Efficiency (XOR)',fontsize=24)

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)