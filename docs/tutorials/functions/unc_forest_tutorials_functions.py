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

from scipy.stats import entropy, norm, multivariate_normal
from scipy.integrate import quad, nquad

import scipy.spatial as ss
from scipy.special import digamma
from math import log
import copy

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

def generate_data_newUF(n, mean, var):
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
    y = np.random.binomial(1, .5, n) # classes are 0 and 1. #UPDATED
    X = np.random.multivariate_normal(mean * y, var * np.eye(n), 1).T # creating the X values using 
    # the randomly distributed y that were generated in the line above

    # from sklearn import datasets
    # from random import sample
    # iris = datasets.load_iris()
    # indices = sample(range(0,150),n)
    # X = iris.data[indices,:2].T
    # y = iris.target[indices]
    
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

def estimate_posterior_newUF(algo, n, mean, var, num_trials, X_eval, parallel = False):
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
        X, y = generate_data_newUF(n, mean, var) # generating data with the function above #UPDATED
        obj.fit(X, y) # using the fit function of the learner to fit the data
        return obj.predict_proba(X_eval)[:,1] # using the predict_proba function on the range of desired X
        
    if parallel:
        predicted_posterior = np.array(Parallel(n_jobs=-2)(delayed(worker)(t) for t in range(num_trials)))
    else:
        predicted_posterior = np.zeros((num_trials, X_eval.shape[0]))
        for t in tqdm(range(num_trials)):
            predicted_posterior[t, :] = worker(t)

    return predicted_posterior

def plot_posterior(ax, algo, num_plotted_trials, X_eval, n, mean, var):
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
    plot_truth(ax, n, mean, var, X_eval)

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


def plot_truth(ax, n, mean, var, X_eval):
    '''
    Parameters
    ---
    ax : list
        Holds the axes of the subplots
    n : int
        The number of data to be generated
    mean : double
        The mean of the data to be generated
    var : double
        The variance in the data to be generated
    X_eval : list
        The range over which to evaluate X values for
    '''
        # By Bayes' rule: (0.5 * X_given_y_1)/((0.5 * X_given_y_1)+(0.5 * X_given_y_negative1))
    #  = (X_given_y_1)/(X_given_y_1+X_given_y_negative1)
    
    # plot ground truth
    opacity = 1
    linewidth = 8
    f_X_given_ypositive = norm.pdf(X_eval.flatten().ravel(), mean, var)
    f_X_given_ynegative = norm.pdf(X_eval.flatten().ravel(), -mean, var)
    ax.plot(X_eval.flatten().ravel(), 
            (f_X_given_ypositive/(f_X_given_ypositive+f_X_given_ynegative)).flatten().ravel(), 
            label = "Truth",
            linewidth = linewidth, 
            color = "black", 
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

def plot_fig1(algos, num_plotted_trials, X_eval, n, mean, var):
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
    n : int
        The number of data to be generated
    mean : double
        The mean of the data to be generated
    var : double
        The variance in the data to be generated
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
                       X_eval, n, mean, var)

    # Create the 1 variance plot. (Rightmost plot in figure 1)
    plot_variance(axes[3], algos, X_eval)
    
    fig.text(0.5, .08, 'x', ha='center') # defining the style of the figure text
    axes[0].set_ylabel(r"$\hat P(Y = 1|X = x)$") # labeling the axes
    axes[0].set_xlabel(" ")
    axes[3].set_ylabel(r"Var($\hat P(Y = 1|X = x)$)")
    
    fig.tight_layout()
    # plt.savefig("fig1.pdf")
    plt.show()

def generate_data_fig2(n, d, mu = 1):
    n_1 = np.random.binomial(n, .5) # number of class 1
    mean = np.zeros(d)
    mean[0] = mu
    X_1 = np.random.multivariate_normal(mean, np.eye(d), n_1)
    
    X = np.concatenate((X_1, np.random.multivariate_normal(-mean, np.eye(d), n - n_1)))
    y = np.concatenate((np.repeat(1, n_1), np.repeat(0, n - n_1)))
  
    return X, y

def cart_estimate(X, y, n_trees = 300, bootstrap = True):
    model = RandomForestClassifier(bootstrap = bootstrap, n_estimators =n_trees)
    model.fit(X, y)
    class_counts = np.zeros((X.shape[0], model.n_classes_))
    for tree_in_forest in model:
        # get number of training elements in each partition
        node_counts = tree_in_forest.tree_.n_node_samples
        # get counts for all x (x.length array)
        partition_counts = np.asarray([node_counts[x] for x in tree_in_forest.apply(X)])
        # get class probability for all x (x.length, n_classes)
        class_probs = tree_in_forest.predict_proba(X)
        # get elements by performing row wise multiplication
        elems = np.multiply(class_probs, partition_counts[:, np.newaxis])
        # update counts for that tree
        class_counts += elems
    probs = class_counts/class_counts.sum(axis=1, keepdims=True)
    entropies = -np.sum(np.log(probs)*probs, axis = 1)
    # convert nan to 0
    entropies = np.nan_to_num(entropies)
    return np.mean(entropies)


def true_cond_entropy(mu, base = np.exp(1)):
    def func(x):
        p = 0.5 * norm.pdf(x, mu, 1) + 0.5 * norm.pdf(x, -mu, 1)
        return -p * np.log(p) / np.log(base)
    
    H_X = quad(func, -20, 20)
    H_XY = 0.5*(1.0 + np.log(2 * np.pi)) / np.log(base)
    H_Y = np.log(2.0) / np.log(base)
    # I_XY = H_X - H_XY = H_Y - H_YX
    return H_Y - H_X[0] + H_XY


def format_func(value, tick_number):
    epsilon = 10 ** (-5)
    if np.absolute(value) < epsilon:
        return "0"
    if np.absolute(value - 0.5) < epsilon:
        return "0.5"
    if np.absolute(value - 1) < epsilon:
        return "1"
    else:
        return ""

def estimate_ce(X, y, label):
    if label == "CART":
        return cart_estimate(X, y)
    elif label == "IRF":
        frac_eval = 0.3
        irf = CalibratedClassifierCV(base_estimator=RandomForestClassifier(n_estimators = 300), 
                                     method='isotonic', 
                                     cv = 5)
        # X_train, y_train, X_eval, y_eval = split_train_eval(X, y, frac_eval)
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=frac_eval)
        irf.fit(X_train, y_train)
        p = irf.predict_proba(X_eval)
        return np.mean(entropy(p.T, base = np.exp(1)))
    elif label == "UF":
        frac_eval = 0.3
        uf = UncertaintyForest(n_estimators = 300, tree_construction_proportion = 0.4, kappa = 3.0)
        # X_train, y_train, X_eval, y_eval = split_train_eval(X, y, frac_eval)
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=frac_eval)
        uf.fit(X_train,y_train)
        p = uf.predict_proba(X_eval)
        return np.mean(entropy(p.T, base = np.exp(1)))
    else:
        raise ValueError("Unrecognized Label!")

def get_cond_entropy_vs_n(mean, d, num_trials, sample_sizes, algos, parallel=False):
    
    def worker(t):
        X, y = generate_data_fig2(elem, d, mu = mean)
        
        ret = []
        for algo in algos:
            ret.append(estimate_ce(X, y, algo['label']))

        return tuple(ret)
    
    output = np.zeros((len(algos), len(sample_sizes), num_trials))
    for i, elem in enumerate(sample_sizes):
        if parallel:
            results = np.array(Parallel(n_jobs=-2)(delayed(worker)(t) for t in range(num_trials)))
        else:
            results = []
            for t in range(num_trials):
                # print(t)
                results.append(worker(t))
            results = np.array(results)

        for j in range(len(algos)):
            output[j, i, :] = results[:, j]
                
    return output

def get_cond_entropy_vs_mu(n, d, num_trials, mus, algos, parallel=False):
    
    def worker(t):
        X, y = generate_data_fig2(n, d, mu = elem)
        
        ret = []
        for algo in algos:
            ret.append(estimate_ce(X, y, algo['label']))

        return tuple(ret)
    
    output = np.zeros((len(algos), len(mus), num_trials))
    for i, elem in enumerate(mus):
        if parallel:
            results = np.array(Parallel(n_jobs=-2)(delayed(worker)(t) for t in range(num_trials)))
        else:
            results = []
            for t in range(num_trials):
                # print(t)
                results.append(worker(t))
            results = np.array(results)

        for j in range(len(algos)):
            output[j, i, :] = results[:, j]
               
    return output

def plot_cond_entropy_by_n(ax, num_plotted_trials, d, mu, algos, panel, num_trials, sample_sizes, parallel=False):
        
    results = get_cond_entropy_vs_n(mu, d, num_trials, sample_sizes, algos, parallel)
    for j, algo in enumerate(algos):
        result = results[j,:,:]

        # Plot the mean over trials as a solid line.
        ax.plot(sample_sizes,
                np.mean(result, axis = 1).flatten(), 
                label = algo['label'], 
                linewidth = 4, 
                color = algo['color'])
        # Use transparent lines to show other trials.
        for t in range(num_plotted_trials):
            ax.plot(sample_sizes, 
                    result[:, t].flatten(),  
                    linewidth = 2, 
                    color = algo['color'],
                    alpha = 0.15)
    
    truth = true_cond_entropy(mu)
    ax.axhline(y = truth, linestyle = '-', color = "black", label = "Truth")
        
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Estimated Conditional Entropy")
    ax.set_title("%s) Effect Size = %.1f" % (panel, mu))
    ax.set_ylim(ymin = -0.05, ymax = 1.05)

def plot_cond_entropy_by_mu(ax, d, n, algos, panel, num_trials, mus, parallel=False):
    
    results = get_cond_entropy_vs_mu(n, d, num_trials, mus, algos, parallel)
    for j, algo in enumerate(algos):
        result = results[j,:,:]

        # Plot the mean over trials as a solid line.
        ax.plot(mus,
                np.mean(result, axis = 1).flatten(), 
                label = algo['label'], 
                linewidth = 4, 
                color = algo['color'])
    
    truth = [true_cond_entropy(mu) for mu in mus]
    ax.plot(mus, truth, label = 'Truth', linewidth = 4, color = 'black')

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.set_ylim(ymin = -.05)
    ax.set_title("%s) n = %d" % (panel, n))
    ax.set_xlabel("Effect Size")
    ax.set_ylabel("Estimated Conditional Entropy")


def plot_fig2(num_plotted_trials, d1, d2, n1, n2, effect_size, algos, num_trials, sample_sizes_d1, sample_sizes_d2, mus, parallel=False):
    sns.set(font_scale = 3)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams['figure.figsize'] = [30, 20]
    fig, axes = plt.subplots(2, 2)
    
    plot_cond_entropy_by_n(axes[0, 0], num_plotted_trials, d1, effect_size, algos, "A", num_trials, sample_sizes_d1, parallel)
    plot_cond_entropy_by_mu(axes[0, 1], d1, n1, algos, "B", num_trials, mus, parallel)
    
    plot_cond_entropy_by_n(axes[1, 0], num_plotted_trials, d2, effect_size, algos, "C", num_trials, sample_sizes_d2, parallel) 
    plot_cond_entropy_by_mu(axes[1, 1], d2, n2, algos, "D", num_trials, mus, parallel)
    
    axes[0,0].legend(loc = "upper left")
    
    fig.text(-0.05, 0.27, 'd = %d' % d2, ha='left', va='center', fontsize = 40)
    fig.text(-0.05, 0.77, 'd = %d' % d1, ha='left', va='center', fontsize = 40)
    
    plt.subplots_adjust(left=-1)
    plt.tight_layout()
    # plt.savefig("fig2.pdf", bbox_inches = "tight")
    plt.show()

def generate_data_fig3(n, d, mu = 1, var1 = 1, pi = 0.5, three_class = False):
    
    means, Sigmas, probs = _make_params(d, mu = mu, var1 = var1, pi = pi, three_class = three_class)
    counts = np.random.multinomial(n, probs, size = 1)[0]
    
    X_data = []
    y_data = []
    for k in range(len(probs)):
        X_data.append(np.random.multivariate_normal(means[k], Sigmas[k], counts[k]))
        y_data.append(np.repeat(k, counts[k]))
    X = np.concatenate(tuple(X_data))
    y = np.concatenate(tuple(y_data))
    
    return X, y

def _make_params(d, mu = 1, var1 = 1, pi = 0.5, three_class = False):
    
    if three_class:
        return _make_three_class_params(d, mu, pi)
    
    mean = np.zeros(d)
    mean[0] = mu
    means = [mean, -mean]

    Sigma1 = np.eye(d)
    Sigma1[0, 0] = var1
    Sigmas = [np.eye(d), Sigma1]
    
    probs = [pi, 1 - pi]
    
    return means, Sigmas, probs

def _make_three_class_params(d, mu, pi):
    
    means = []
    mean = np.zeros(d)
    
    mean[0] = mu
    means.append(copy.deepcopy(mean))
    
    mean[0] = -mu
    means.append(copy.deepcopy(mean))
    
    mean[0] = 0
    mean[d-1] = mu
    means.append(copy.deepcopy(mean))
    
    Sigmas = [np.eye(d)]*3
    probs = [pi, (1 - pi) / 2, (1 - pi) / 2]
    
    return means, Sigmas, probs

def plot_setting(n, setting, ax):
    
    mean = 3 if setting['name'] == 'Three Class Gaussians' else 1
    X, y = generate_data_fig3(n, 2, **setting['kwargs'], mu = mean)
        
    colors = ["#c51b7d", "#2166ac", "#d95f02"]
    ax.scatter(X[:, 0], X[:, 1], color = np.array(colors)[y], marker = ".")
    
    ax.set_xlim(left = -5.05)
    ax.set_xlim(right = 5.05)
    
    ax.set_ylabel(setting['name'])

def compute_mutual_info(d, base = np.exp(1), mu = 1, var1 = 1, pi = 0.5, three_class = False):
    
    if d > 1:
        dim = 2
    else:
        dim = 1
 
    means, Sigmas, probs = _make_params(dim, mu = mu, var1 = var1, pi = pi, three_class = three_class)
    
    # Compute entropy and X and Y.
    def func(*args):
        x = np.array(args)
        p = 0
        for k in range(len(means)):
            p += probs[k] * multivariate_normal.pdf(x, means[k], Sigmas[k])
        return -p * np.log(p) / np.log(base)

    scale = 10
    lims = [[-scale, scale]]*dim
    H_X, int_err = nquad(func, lims)
    H_Y = entropy(probs, base = base)
    
    # Compute MI.
    H_XY = 0
    for k in range(len(means)):
        H_XY += probs[k] * (dim * np.log(2*np.pi) + np.log(np.linalg.det(Sigmas[k])) + dim) / (2 * np.log(base))
    I_XY = H_X - H_XY
    
    return I_XY, H_X, H_Y

def estimate_mi(X, y, label, est_H_Y, norm_factor):
    
    if label == "IRF":
        frac_eval = 0.3
        irf = CalibratedClassifierCV(base_estimator=RandomForestClassifier(n_estimators = 60), 
                                     method='isotonic', 
                                     cv = 5)
        # X_train, y_train, X_eval, y_eval = split_train_eval(X, y, frac_eval)
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=frac_eval)
        irf.fit(X_train, y_train)
        p = irf.predict_proba(X_eval)
        return (est_H_Y - np.mean(entropy(p.T, base = np.exp(1)))) / norm_factor
    elif label == "UF":
        frac_eval = 0.3
        uf = UncertaintyForest(n_estimators = 300, tree_construction_proportion = 0.4, kappa = 3.0)
        # X_train, y_train, X_eval, y_eval = split_train_eval(X, y, frac_eval)
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=frac_eval)
        uf.fit(X_train, y_train)
        p = uf.predict_proba(X_eval)
        return (est_H_Y - np.mean(entropy(p.T, base = np.exp(1)))) / norm_factor
    elif label == "KSG":
        return ksg(X, y.reshape(-1, 1)) / norm_factor
    elif label == "Mixed KSG":
        return mixed_ksg(X, y.reshape(-1, 1)) / norm_factor
    else:
        raise ValueError("Unrecognized Label!")

def get_plot_mutual_info_by_pi(setting, algos, d, ax, n, pis, num_trials, parallel=False):
    def worker(t):
        X, y = generate_data_fig3(n, d, pi = elem, **setting['kwargs'])
        
        I_XY, H_X, H_Y = compute_mutual_info(d, pi = elem, **setting['kwargs'])
        norm_factor = min(H_X, H_Y)
        
        _, counts = np.unique(y, return_counts=True)
        est_H_Y = entropy(counts, base=np.exp(1))
        
        ret = []
        for algo in algos:
            ret.append(estimate_mi(X, y, algo['label'], est_H_Y, norm_factor))

        return tuple(ret)

    output = np.zeros((len(algos), len(pis), num_trials))
    for i, elem in enumerate(pis):
        if parallel:
            results = np.array(Parallel(n_jobs=-2)(delayed(worker)(t) for t in range(num_trials)))
            
        else:
            results = []
            for t in range(num_trials):
                # print(t)
                results.append(worker(t))
            results = np.array(results)

        for j in range(len(algos)):
            output[j, i, :] = results[:, j]


    for j, algo in enumerate(algos):
        result = output[j,:,:]
        # Plot the mean over trials as a solid line.
        ax.plot(pis,
                np.mean(result, axis = 1).flatten(), 
                label = algo['label'], 
                linewidth = 4, 
                color = algo['color'])
        
    # ax.set_yscale('log')

    truth = np.zeros(len(pis))
    for i, pi in enumerate(pis):
        I_XY, H_X, H_Y = compute_mutual_info(d, pi = pi, **setting['kwargs'])
        truth[i] = I_XY / min(H_X, H_Y)
    
    ax.plot(pis, truth, label = 'Truth', linewidth = 2, color = 'black')

    ax.set_xlabel("Class Prior")
    ax.set_xlim((np.amin(pis) - 0.05, np.amax(pis) + 0.05))
    ax.set_ylim((-0.05, 0.55))
    ax.set_ylabel("Estimated Normalized MI")

def get_plot_mutual_info_by_d(setting, algos, mu, ax, n, ds, num_trials, parallel=False):

    def worker(t):
        X, y = generate_data_fig3(n, elem, mu = mu, **setting['kwargs'])
        
        I_XY, H_X, H_Y = compute_mutual_info(elem, mu = mu, **setting['kwargs'])
        norm_factor = min(H_X, H_Y)
        
        _, counts = np.unique(y, return_counts=True)
        est_H_Y = entropy(counts, base=np.exp(1))
        
        ret = []
        for algo in algos:
            ret.append(estimate_mi(X, y, algo['label'], est_H_Y, norm_factor))

        return tuple(ret)
    
    output = np.zeros((len(algos), len(ds), num_trials))
    for i, elem in enumerate(ds):
        if parallel:
            results = np.array(Parallel(n_jobs=-2)(delayed(worker)(t) for t in range(num_trials)))
            
        else:
            results = []
            for t in range(num_trials):
                # print(t)
                results.append(worker(t))
            results = np.array(results)
        
        for j in range(len(algos)):
            output[j, i, :] = results[:, j]

    for j, algo in enumerate(algos):
        result = output[j,:,:]
        # Plot the mean over trials as a solid line.
        ax.plot(ds,
                np.mean(result, axis = 1).flatten(), 
                label = algo['label'], 
                linewidth = 4, 
                color = algo['color'])

    I_XY, H_X, H_Y = compute_mutual_info(2, **setting['kwargs'], mu = mu)
    truth = np.repeat(I_XY / min(H_X, H_Y), len(ds))
    ax.plot(ds, truth, label = 'Truth', linewidth = 2, color = 'black')

    ax.set_xlabel("Dimensionality")
    ax.set_xlim(left = np.amin(ds) - 0.05)
    ax.set_xlim(right = np.amax(ds) + 0.05)
    ax.set_ylim((-0.05, 0.55))
    ax.set_ylabel("Estimated Normalized MI")

def plot_fig3(algos, n, d, mu, settings, pis, ds, num_trials, parallel=False):
    sns.set(font_scale = 1.5)
    sns.set_style("ticks")
    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(len(settings), 3, figsize = (15,13))

    for s, setting in enumerate(settings):
        plot_setting(2000, setting, axes[s, 0])
        get_plot_mutual_info_by_pi(setting, algos, d, axes[s, 1], n, pis, num_trials, parallel)
        get_plot_mutual_info_by_d(setting, algos, mu, axes[s, 2], n, ds, num_trials, parallel)
        
    axes[0, 1].set_title('n = %d, d = %d' % (n, d))
    axes[0, 2].set_title('n = %d, Effect Size = %.1f' % (n, mu))
    axes[2, 2].legend(loc = "lower right")

    plt.yticks(fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.tight_layout()
    # plt.savefig("fig3.pdf")
    plt.show()

def mixed_ksg(x, y, k=5):
    """
    Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
    Using *Mixed-KSG* mutual information estimator
    Input: x: 2D array of size N*d_x (or 1D list of size N if d_x = 1)
    y: 2D array of size N*d_y (or 1D list of size N if d_y = 1)
    k: k-nearest neighbor parameter
    Output: one number of I(X;Y)
    """

    assert len(x) == len(y), "Lists should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"

    N = len(x)
    if x.ndim == 1:
        x = x.reshape((N, 1))

    # dx = len(x[0])

    if y.ndim == 1:
        y = y.reshape((N, 1))
    # dy = len(y[0])
    data = np.concatenate((x, y), axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point, k + 1, p=float("inf"))[0][k] for point in data]
    ans = 0

    for i in range(N):
        kp, nx, ny = k, k, k
        if knn_dis[i] == 0:
            kp = len(tree_xy.query_ball_point(data[i], 1e-15, p=float("inf")))
            nx = len(tree_x.query_ball_point(x[i], 1e-15, p=float("inf")))
            ny = len(tree_y.query_ball_point(y[i], 1e-15, p=float("inf")))
        else:
            nx = len(tree_x.query_ball_point(x[i], knn_dis[i] - 1e-15, p=float("inf")))
            ny = len(tree_y.query_ball_point(y[i], knn_dis[i] - 1e-15, p=float("inf")))
        ans += (digamma(kp) + log(N) - digamma(nx) - digamma(ny)) / N
    return ans

# Original KSG estimator (Blue line)
def ksg(x, y, k=5):
    assert len(x) == len(y), "Lists should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    N = len(x)
    if x.ndim == 1:
        x = x.reshape((N, 1))
    # dx = len(x[0])
    if y.ndim == 1:
        y = y.reshape((N, 1))
    # dy = len(y[0])
    data = np.concatenate((x, y), axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point, k + 1, p=float("inf"))[0][k] for point in data]
    ans = 0

    for i in range(N):
        nx = len(tree_x.query_ball_point(x[i], knn_dis[i] + 1e-15, p=float("inf"))) - 1
        ny = len(tree_y.query_ball_point(y[i], knn_dis[i] + 1e-15, p=float("inf"))) - 1
        ans += (digamma(k) + log(N) - digamma(nx) - digamma(ny)) / N
    return ans