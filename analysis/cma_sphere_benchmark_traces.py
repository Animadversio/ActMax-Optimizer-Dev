import nevergrad as ng
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from nevergrad.optimization.callbacks import ParametersLogger
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as pkl
from os.path import join
from glob import glob
mpl.rcParams['pdf.fonttype'] = 42 # set font for export to pdfs
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
figdir = r"E:\OneDrive - Harvard University\GECCO2022\Figures\CMABenchmark"
#%%
unitdir = r"E:\Cluster_Backup\cma_optim_cmp\alexnet_.classifier.Linear6_001"
pklpaths = glob(join(unitdir, "*.pkl"))
df_all_col = []
for fpath in pklpaths:
    # data = pkl.load(open(join(unitdir, "summarize_alexnet_.classifier.Linear6_001_89157.pkl"), "rb"))
    data = pkl.load(open(fpath, "rb"))
    if "CholeskyCMAES" not in data:
        continue
    #% Curate them into a collection of dataFrames
    df_all = {}
    for optim in data:
        df_all[optim] = pd.DataFrame()
        df_all[optim]["scores"] = data[optim]['cleanscores_all']
        if optim in ["Genetic", "pycma", "pycmaDiagonal"]:
            generations = np.repeat(np.arange(1, 75+1), 40)
        elif optim in ["CholeskyCMAES"]:
            generations = np.hstack((np.array([1]), np.repeat(np.arange(2, 75+1), 40)))
        elif optim in ["ZOHA_Sphere_exp", "ZOHA_Sphere_inv"]:
            generations = np.hstack((np.repeat([1],40), np.repeat(np.arange(2, 75+1), 41)))
        df_all[optim]["generations"] = generations
    df_all_col.append(df_all)
#%%
import seaborn as sns

colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
scatter = True
cutoff = 3000
plt.figure(figsize=[6, 6])
# for k, df_all in enumerate(df_all_col):
df_all = df_all_col[3]
for i, optim in enumerate(df_all):
    if scatter:
        plt.scatter(np.arange(1, min(cutoff,len(df_all[optim]["scores"]))+1), df_all[optim]["scores"][:cutoff],
             s=4, alpha=0.2, label=None, color=colorcycle[i])
    plt.plot(np.arange(1, min(cutoff,len(df_all[optim]["scores"]))+1), df_all[optim]["scores"].cummax()[:cutoff],
             lw=2, alpha=0.8, label=optim, color=colorcycle[i])

plt.xlabel("Function Evaluation", fontsize=16)
plt.ylabel("Activation", fontsize=16)
plt.title("Score Traces Comparison\nResNet50-Robust Linear Unit 1", fontsize=18)
plt.legend(fontsize=14)
plt.savefig(join(figdir, r"example_score_traces%s.png" % ("_w_scatter" if scatter else "")))
plt.savefig(join(figdir, r"example_score_traces%s.pdf" % ("_w_scatter" if scatter else "")))
plt.show()
#%%
colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
scatter = True
plt.figure(figsize=[6, 6])
for k, df_all in enumerate(df_all_col):
    for i, optim in enumerate(df_all):
        if scatter and k == 0:
            plt.scatter(np.arange(1, len(df_all[optim]["scores"])+1), df_all[optim]["scores"],
                 s=4, alpha=0.1, label=None, color=colorcycle[i])
        plt.plot(np.arange(1, len(df_all[optim]["scores"])+1), df_all[optim]["scores"].cummax(),
                 lw=2, alpha=1, label=optim if k == 0 else None, color=colorcycle[i])

plt.xlabel("Function Evaluation", fontsize=16)
plt.ylabel("Activation", fontsize=16)
plt.title("Score Traces Comparison\nResNet50-Robust Linear Unit 1", fontsize=18)
plt.legend(fontsize=14)
plt.savefig(join(figdir, r"example_score_traces-multi%s.png" % ("_w_scatter" if scatter else "")))
plt.savefig(join(figdir, r"example_score_traces-multi%s.pdf" % ("_w_scatter" if scatter else "")))
plt.show()
#%%
legend = False
figh, axs = plt.subplots(2, 3, figsize=[7.5, 6])
for k, df_all in enumerate(df_all_col):
    for i, optim in enumerate(df_all):
        plt.sca(axs.flatten()[i],)
        if scatter and k == 2:
            sns.scatterplot(x=df_all[optim]["generations"], y=df_all[optim]["scores"],
                    s=12, alpha=0.15, **({"legend":None} if ~legend else {}))
        sns.lineplot(x=df_all[optim]["generations"], y=df_all[optim]["scores"], label=optim,
                     color="red", **({"legend":None} if ~legend else {}))
        plt.title(optim)
        plt.xlabel(None)
        plt.ylabel(None)
figh.suptitle("Score Traces Comparison\nResNet50-Robust Linear Unit 1", fontsize=16)
plt.savefig(join(figdir, r"example_montage_score_traces_indiv-multi%s.png" % ("_w_scatter" if scatter else "")))
plt.savefig(join(figdir, r"example_montage_score_traces_indiv-multi%s.pdf" % ("_w_scatter" if scatter else "")))
plt.show()