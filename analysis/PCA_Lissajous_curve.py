import os
import re
from os.path import join
from glob import glob
from time import time
from easydict import EasyDict
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 200

figdir = "E:\OneDrive - Harvard University\GECCO2022\Figures\Sinusoidal"

#%% Plot Example Lissajour curves
import matplotlib.collections as mcoll
import matplotlib.path as mpath
def colorline(
    x, y, z=None, ax=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_Lissajous(projtraj, K=6, unitlabel="", figdir="", figsize=(8, 8)):
    figh, axs = plt.subplots(K, K, figsize=figsize, )
    for i in range(K + 1):
        for j in range(i):
            colorline(projtraj[:, j], projtraj[:, i], ax=axs[i - 1, j],
                      cmap=plt.get_cmap('jet'), linewidth=2, alpha=0.6)
            # timearr = np.arange(len(projtraj[:,j]))
            # axs[i-1, j].plot(projtraj[:,j], projtraj[:,i], c="k")
            # timearr = np.arange(len(projtraj[:,j]))
            # axs[i-1, j].scatter(projtraj[:,j], projtraj[:,i], c=timearr, marker='o',)
            if j > 0:
                axs[i - 1, j].set_yticklabels([])
            if j == 0:
                axs[i - 1, j].set_ylabel(f"PC {i + 1}")
            if i < K:
                axs[i - 1, j].set_xticklabels([])
            if i == K:
                axs[i - 1, j].set_xlabel(f"PC {j + 1}")
            axs[i - 1, j].axis("auto")

    for i in range(K):
        for j in range(i + 1, K):
            axs[i, j].axis("off")
    figh.tight_layout()
    plt.savefig(join(figdir, f"Example_Lissajous_curves_{unitlabel}_time.png"))
    plt.savefig(join(figdir, f"Example_Lissajous_curves_{unitlabel}_time.pdf"))
    plt.show()
    return figh, axs

#%% Load data
unitlabel = "alexnet_.classifier.Linear6_000"
unitdir = join(r"E:\Cluster_Backup\cma_optim_cmp", unitlabel)
trajdata = np.load(join(unitdir, "CholeskyCMAES_%s_12731.npz"%unitlabel))
meanPCdata = np.load(join(unitdir, "CholeskyCMAES_meanPCA_coef_%s_12731.npz"%unitlabel))
PCdata = np.load(join(unitdir, "CholeskyCMAES_PCA_coef_%s_12731.npz"%unitlabel))
#%%
projtraj = meanPCdata['PCcoefs_mean']
plot_Lissajous(projtraj, K=6, unitlabel="fc8_01", figdir=figdir, figsize=(8, 8))
#%%
import numpy as np
from core.insilico_exps import ExperimentEvolution
from core.Optimizers import CholeskyCMAES, ZOHA_Sphere_lr_euclid, Genetic, pycma_optimizer
from time import time
tmpsavedir = "E:\\" # Temporary save directory
# load optimizer
optim = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20, lr=1.5, sphere_norm=300)
optim.lr_schedule(n_gen=75, mode="exp", lim=(50, 7.33) ,)
explabel, model_unit = "alexnet_fc8_1_sphere", ("alexnet", ".classifier.Linear6", 1)
Exp = ExperimentEvolution(model_unit, max_step=75, savedir=tmpsavedir, explabel=explabel, optimizer=optim)
t1 = time()
Exp.run(optim.get_init_pop())
print(time() - t1, "sec")  # 10.4 sec for 75 generations! 55.026 +- std 1.700
#%%
codes_mean = np.array([Exp.codes_all[Exp.generations == geni, :].mean(axis=0) for geni in range(75)])
PCcoefs_mean_sphere = PCA(n_components=60).fit_transform(codes_mean)
plot_Lissajous(PCcoefs_mean_sphere, K=6, unitlabel="fc8_01_sphere_PCmean", figdir=figdir, figsize=(8, 8))
#%%
PCcoefs_all_sphere = PCA(n_components=60).fit(Exp.codes_all).transform(codes_mean)
plot_Lissajous(PCcoefs_all_sphere, K=6, unitlabel="fc8_01_sphere_PCall", figdir=figdir, figsize=(8, 8))
#%%
optim = CholeskyCMAES(4096, population_size=40, init_sigma=2.0, Aupdate_freq=10, init_code=np.zeros([1, 4096]))
explabel, model_unit = "alexnet_fc8_1", ("alexnet", ".classifier.Linear6", 1)
Exp2 = ExperimentEvolution(model_unit, max_step=75, savedir=tmpsavedir, explabel=explabel, optimizer=optim)
t1 = time()
Exp2.run(optim.get_init_pop())
print(time() - t1, "sec")
#%%
codes_mean = np.array([Exp2.codes_all[Exp2.generations == geni, :].mean(axis=0) for geni in range(75)])
PCcoefs_mean = PCA(n_components=60).fit_transform(codes_mean)
plot_Lissajous(PCcoefs_mean, K=6, unitlabel="fc8_01_PCmean", figdir=figdir, figsize=(8, 8))
#%%
PCcoefs_all = PCA(n_components=60).fit(Exp2.codes_all).transform(codes_mean)
plot_Lissajous(PCcoefs_all, K=6, unitlabel="fc8_01_PCall", figdir=figdir, figsize=(8, 8))
