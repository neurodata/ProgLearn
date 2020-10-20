import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from tqdm.notebook import tqdm
from joblib import Parallel, delayed

from proglearn.forest import UncertaintyForest
from proglearn.sims import generate_gaussian_parity

def generate_data(n, mean, var):
    '''
    Parameters
    ---
    n : int
        The number of data to be generated
    mean : double
        The mean of the data to be generated
    var : double
        The variance in the data to be generated
    '''
    y = 2 * np.random.binomial(1, .5, n) - 1 # classes are -1 and 1.
    X = np.random.multivariate_normal(mean * y, var * np.eye(n), 1).T # creating the X values using 
    # the randomly distributed y that were generated in the line above
    
    return X, y

def estimate_posterior(algo, n, mean, var, num_trials, X_eval, parallel = False):
    '''
    Estimate posteriors for many trials and evaluate in the given X_eval range
    
    Parameters
    ---
    algo : dict
        A dictionary of the learner to be used containing a key "instance" of the learner
    n : int
        The number of data to be generated
    mean : double
        The mean of the data used
    var : double
        The variance of the data used
    num_trials : int
        The number of trials to run over
    X_eval : list
        The range over which to evaluate X values for
    '''
    obj = algo['instance'] # grabbing the instance of the learner 
    def worker(t):
        X, y = generate_data(n, mean, var) # generating data with the function above
        obj.fit(X, y) # using the fit function of the learner to fit the data
        return obj.predict_proba(X_eval)[:,1] # using the predict_proba function on the range of desired X
        
    if parallel:
        predicted_posterior = np.array(Parallel(n_jobs=-2)(delayed(worker)(t) for t in range(num_trials)))
    else:
        predicted_posterior = np.zeros((num_trials, X_eval.shape[0]))
        for t in tqdm(range(num_trials)):
            predicted_posterior[t, :] = worker(t)

    return predicted_posterior

def plot_posterior(ax, algo, num_plotted_trials, X_eval):
    """
    Will be used for CART, Honest, or Uncertainty Forest to plot P(Y = 1 | X = x). 
    This is the left three plots in figure 1.
    Plots each of num_plotted_trials iterations, highlighting a single line
    
    Parameters
    ---
    ax : list
        Holds the axes of the subplots
    algo : dict
        A dictionary of the learner to be used containing a key "instance" of the learner
    num_plotted_trials : int
        The number of trials that will be overlayed. This is shown as the lighter lines figure 1.
    X_eval : list
        The range over which to evaluate X values for
    """
    for i in range(num_plotted_trials):
        linewidth = 1
        opacity = .3
        if i == num_plotted_trials - 1:
            opacity = 1
            linewidth = 8
        ax.set_title(algo['title'])
        ax.plot(X_eval.flatten().ravel(), algo['predicted_posterior'][i, :].ravel(), 
                label = algo['label'],
                linewidth = linewidth, 
                color = algo['color'], 
                alpha = opacity)


def plot_variance(ax, algos, X_eval):
    """
    Will be used for the rightmost plot in figure 1.
    Plots the variance over the number of trials.
    
    Parameters
    ---
    ax : list
        Holds the axes of the subplots
    algos : list
        A list of dictionaries of the learners to be used
    X_eval : list
        The range over which to evaluate X values for
    """
    ax.set_title('Posterior Variance') # adding a title to the plot
    for algo in algos: # looping over the algorithms used
        variance = np.var(algo['predicted_posterior'], axis = 0) # determining the variance
        ax.plot(X_eval.flatten().ravel(), variance.ravel(), 
                label = algo['label'],
                linewidth = 8, 
                color = algo['color']) # plotting

def plot_fig1(algos, num_plotted_trials, X_eval):
    """
    Sets the communal plotting parameters and creates figure 1

    Parameters
    ---
    algos : list
        A list of dictionaries of the learners to be used
    num_plotted_trials : int
        The number of trials that will be overlayed. This is shown as the lighter lines figure 1.
    X_eval : list
        The range over which to evaluate X values for
    """
    sns.set(font_scale = 6) # setting font size
    sns.set_style("ticks") # setting plot style
    plt.rcParams['figure.figsize'] = [55, 14] # setting figure size
    fig, axes = plt.subplots(1, 4) # creating the axes (that will be passed to the subsequent functions)
    for ax in axes[0:3]:
        ax.set_xlim(-2.1, 2.1) # setting x limits
        ax.set_ylim(-0.05, 1.05) # setting y limits

    # Create the 3 posterior plots. (Left three plots in figure 1)
    for i in range(len(algos)):
        plot_posterior(axes[i], 
                       algos[i],
                       num_plotted_trials, 
                       X_eval)

    # Create the 1 variance plot. (Rightmost plot in figure 1)
    plot_variance(axes[3], algos, X_eval)
    
    fig.text(0.5, .08, 'x', ha='center') # defining the style of the figure text
    axes[0].set_ylabel(r"$\hat P(Y = 1|X = x)$") # labeling the axes
    axes[0].set_xlabel(" ")
    axes[3].set_ylabel(r"Var($\hat P(Y = 1|X = x)$)")
    
    fig.tight_layout()
    # plt.savefig("fig1.pdf")
    plt.show()