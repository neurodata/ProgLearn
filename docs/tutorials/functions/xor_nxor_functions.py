import numpy as np
import random

import seaborn as sns 
import matplotlib.pyplot as plt

from proglearn.forest import LifelongClassificationForest, UncertaintyForest
from proglearn.sims import *


def get_colors(colors, inds):
    c = [colors[i] for i in inds]
    return c

def plot_xor_nxor(data, labels, title):
    colors = sns.color_palette('Dark2', n_colors=2)
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.scatter(data[:, 0], data[:, 1], c=get_colors(colors, labels), s=50)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=30)
    plt.tight_layout()
    ax.axis('off')
    plt.show()

def experiment(n_xor, n_nxor, n_test, reps, n_trees, max_depth, acorn=None):
    """
    Runs the Gaussian XOR N-XOR experiment.
    Returns the mean error.
    """

    # initialize experiment
    if n_xor==0 and n_nxor==0:
        raise ValueError('Wake up and provide samples to train!!!')
    
    # if acorn is specified, set random seed to it
    if acorn != None:
        np.random.seed(acorn)
    
    # initialize array for storing errors
    errors = np.zeros((reps,4),dtype=float)
    
    # run the progressive learning algorithm for a number of repetitions
    for i in range(reps):
        
        # initialize learners
        progressive_learner = LifelongClassificationForest(n_estimators=n_trees)
        uf = UncertaintyForest(n_estimators=2*n_trees)
        
        # source data
        xor, label_xor = generate_gaussian_parity(n_xor,angle_params=0)
        test_xor, test_label_xor = generate_gaussian_parity(n_test,angle_params=0)
    
        # target data
        nxor, label_nxor = generate_gaussian_parity(n_nxor,angle_params=np.pi/2)
        test_nxor, test_label_nxor = generate_gaussian_parity(n_test,angle_params=np.pi/2)
    
        if n_xor == 0:
            # fit learners and predict
            progressive_learner.add_task(nxor, label_nxor)
            l2f_task2=progressive_learner.predict(test_nxor, task_id=0)
            uf.fit(nxor, label_nxor)
            uf_task2=uf.predict(test_nxor)
            # record errors
            errors[i,0] = 0.5 # no data, so random chance of guessing correctly (err = 0.5)
            errors[i,1] = 0.5 # no data, so random chance of guessing correctly (err = 0.5)
            errors[i,2] = 1 - np.sum(uf_task2 == test_label_nxor)/n_test
            errors[i,3] = 1 - np.sum(l2f_task2 == test_label_nxor)/n_test
        elif n_nxor == 0:
            # fit learners and predict
            progressive_learner.add_task(xor, label_xor)
            l2f_task1=progressive_learner.predict(test_xor, task_id=0)
            uf.fit(xor, label_xor)
            uf_task1=uf.predict(test_xor)
            # record errors
            errors[i,0] = 1 - np.sum(uf_task1 == test_label_xor)/n_test
            errors[i,1] = 1 - np.sum(l2f_task1 == test_label_xor)/n_test
            errors[i,2] = 0.5 # no data, so random chance of guessing correctly (err = 0.5)
            errors[i,3] = 0.5 # no data, so random chance of guessing correctly (err = 0.5)
        else:
            # fit learners and predict
            progressive_learner.add_task(xor, label_xor)
            progressive_learner.add_task(nxor, label_nxor)
            l2f_task1=progressive_learner.predict(test_xor, task_id=0)
            l2f_task2=progressive_learner.predict(test_nxor, task_id=1)
            uf.fit(xor, label_xor)
            uf_task1=uf.predict(test_xor)
            uf.fit(nxor, label_nxor)
            uf_task2=uf.predict(test_nxor)
            # record errors
            errors[i,0] = 1 - np.sum(uf_task1 == test_label_xor)/n_test
            errors[i,1] = 1 - np.sum(l2f_task1 == test_label_xor)/n_test
            errors[i,2] = 1 - np.sum(uf_task2 == test_label_nxor)/n_test
            errors[i,3] = 1 - np.sum(l2f_task2 == test_label_nxor)/n_test
    
    return np.mean(errors,axis=0)
    
def plot_error(x, y1, y2, ls, task):
    # define labels
    algorithms = ['Uncertainty Forest', 'Lifelong Forest']
    TASK1='XOR'
    TASK2='N-XOR'
    colors = sns.color_palette("Set1", n_colors = 2)
    # plot and format
    fig1 = plt.figure(figsize=(8,8))
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(x, y1, label=algorithms[0], c=colors[1], ls=ls, lw=3)
    ax1.plot(x, y2, label=algorithms[1], c=colors[0], ls=ls, lw=3)
    ax1.legend(loc='upper right', fontsize=24, frameon=False)
    ax1.set_xlabel('Total Sample Size', fontsize=35)
    ax1.set_ylabel('Generalization Error (%s)'%(task), fontsize=35)
    ax1.tick_params(labelsize=27.5)
    ax1.set_xticks([250,750,1500])
    ax1.set_yticks([0.05, 0.10])
    ax1.set_xlim(-10)
    ax1.set_ylim(0.03, 0.15)
    ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
    right_side = ax1.spines["right"]
    right_side.set_visible(False)
    top_side = ax1.spines["top"]
    top_side.set_visible(False)
    ax1.text(400, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=30)
    ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=30)
    plt.tight_layout()
    plt.show()
    
def plot_eff(x1, x2, y1, y2):
    # define labels
    algorithms = ['Forward Transfer', 'Backward Transfer']
    TASK1='XOR'
    TASK2='N-XOR'
    colors = sns.color_palette("Set1", n_colors = 2)
    # plot and format
    fig1 = plt.figure(figsize=(8,8))
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(x1, y1, label=algorithms[0], c=colors[0], ls='-', lw=3)
    ax1.plot(x2, y2, label=algorithms[1], c=colors[0], ls='--', lw=3)
    ax1.set_ylabel('Transfer Efficiency', fontsize=35)
    ax1.legend(loc='upper right', fontsize=24, frameon=False)
    ax1.set_ylim(0.95, 1.42)
    ax1.set_xlabel('Total Sample Size', fontsize=35)
    ax1.tick_params(labelsize=27.5)
    ax1.set_yticks([1, 1.4])
    ax1.set_xticks([250,750,1500])
    ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
    right_side = ax1.spines["right"]
    right_side.set_visible(False)
    top_side = ax1.spines["top"]
    top_side.set_visible(False)
    ax1.hlines(1, 50,1500, colors='gray', linestyles='dashed',linewidth=1.5)
    ax1.text(400, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=30)
    ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=30)
    plt.tight_layout()