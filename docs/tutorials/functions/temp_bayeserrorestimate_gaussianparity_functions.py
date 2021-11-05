#Plotting Gaussians created by the function
import seaborn as sns
import matplotlib.pyplot as plt

def get_colors(colors, inds):
    c = [colors[i] for i in inds]
    return c


def plot_gaussians(Values, Classes):
    X = Values
    y = Classes
    colors = sns.color_palette("Dark2", n_colors=2)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(X[:, 0], X[:, 1], c=get_colors(colors, y), s=50)
    ax.set_title("Created Gaussians", fontsize=30)
    plt.tight_layout()
