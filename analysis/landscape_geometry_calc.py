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
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 200
figdir = r"E:\OneDrive - Harvard University\GECCO2022\Figures\TuneGeometry"
#%%
datadir = r"E:\OneDrive - Washington University in St. Louis\HessTune\AlexNet"
data = np.load(join(datadir, "fc8_1_(1_1).npz"))
#%%
for x in data:
    print(x, data[x].shape)
#%%
plt.semilogy(sorted(np.abs(data["eigvals_u"])))
plt.semilogy(sorted(np.abs(data["eigvals"])))
plt.show()
#%% Visualize and plot the eigenvalues
unitfnlist = ["conv1_0_(27_27).npz",
            "conv2_0_(13_13).npz",
            "conv3_0_(6_6).npz",
            "conv4_0_(6_6).npz",
            "conv5_0_(6_6).npz",
            "fc6_0_(1_1).npz",
            "fc7_0_(1_1).npz",
            "fc8_0_(1_1).npz",]

def plot_CNN_spect_comparison(maxnorm=True, colorcyc=plt.rcParams['axes.prop_cycle'].by_key()['color']):
    figh = plt.figure(figsize=[6, 6])
    for k, unitfn in enumerate(unitfnlist):
        match = re.findall("(.*)_(\d*)_\((\d*)_(\d*)\).npz", unitfn, )
        layer, channum, pos = match[0][0], int(match[0][1]), (int(match[0][2]), int(match[0][3]))
        for chi in range(5):
            unitfn = "%s_%d_(%d_%d).npz" % (layer, chi, pos[0], pos[1])
            data = np.load(join(datadir, unitfn))
            eigvals = data["eigvals"].copy()
            normalizer = np.abs(eigvals).max() if maxnorm else np.array(1.0)
            plt.semilogy(sorted(np.abs(eigvals))[::-1] / normalizer,
                         label=layer if chi == 0 else None,
                         lw=2, alpha=0.5, color=colorcyc[k])
    if maxnorm:
        figh.gca().set_ylim(1E-9, 1.5)
    else:
        figh.gca().set_ylim(1E-9, 10)

    plt.ylabel("abs Hessian eigenvalue"+"(norm by max)" if maxnorm else "",fontsize=12)
    plt.xlabel("eigen rank",fontsize=12)
    plt.title("Hessian of Landscape Compared across CNN Hierarchy\nEvaluated at peak of act max",fontsize=14)
    plt.legend()
    plt.savefig(join(figdir, "alexnet_layers_peak_hess_spect_cmp%s%s.png"%("_maxnorm" if maxnorm else "", "")))
    plt.savefig(join(figdir, "alexnet_layers_peak_hess_spect_cmp%s%s.pdf"%("_maxnorm" if maxnorm else "", "")))
    plt.show()
    return figh

cmap = plt.get_cmap("hsv")
colorcyc = [cmap(i/(8)) for i in range(8)]
plot_CNN_spect_comparison(maxnorm=True, colorcyc=colorcyc)
plot_CNN_spect_comparison(maxnorm=False, colorcyc=colorcyc)
#%%
def plot_CNN_spect_comparison_randinit(maxnorm=True, colorcyc=plt.rcParams['axes.prop_cycle'].by_key()['color']):
    figh = plt.figure(figsize=[6, 6])
    for k, unitfn in enumerate(unitfnlist):
        match = re.findall("(.*)_(\d*)_\((\d*)_(\d*)\).npz", unitfn, )
        layer, channum, pos = match[0][0], int(match[0][1]), (int(match[0][2]), int(match[0][3]))
        for chi in range(5):
            unitfn = "%s_%d_(%d_%d).npz" % (layer, chi, pos[0], pos[1])
            data = np.load(join(datadir, unitfn))
            eigvals = data["eigvals_u"].copy()
            normalizer = np.abs(eigvals).max() if maxnorm else np.array(1.0)
            plt.semilogy(sorted(np.abs(eigvals))[::-1] / normalizer,
                         label=layer if chi == 0 else None,
                         lw=2, alpha=0.5, color=colorcyc[k])
    if maxnorm:
        figh.gca().set_ylim(3E-8, 1.5)
    else:
        figh.gca().set_ylim(1E-8, 10)

    plt.ylabel("abs Hessian eigenvalue"+"(norm by max)" if maxnorm else "",fontsize=12)
    plt.xlabel("eigen rank",fontsize=12)
    plt.title("Hessian of Landscape Compared across CNN Hierarchy\nEvaluated at random vector",fontsize=14)
    plt.legend()
    plt.savefig(join(figdir, "alexnet_layers_rnd_hess_spect_cmp%s%s.png"%("_maxnorm" if maxnorm else "", "")))
    plt.savefig(join(figdir, "alexnet_layers_rnd_hess_spect_cmp%s%s.pdf"%("_maxnorm" if maxnorm else "", "")))
    plt.show()
    return figh
#%%
plot_CNN_spect_comparison_randinit(maxnorm=True, colorcyc=colorcyc)
plot_CNN_spect_comparison_randinit(maxnorm=False, colorcyc=colorcyc)
