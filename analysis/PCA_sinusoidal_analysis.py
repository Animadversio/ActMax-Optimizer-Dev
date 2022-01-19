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
import matplotlib
import pickle as pkl
from sklearn.decomposition import PCA
matplotlib.rcParams['pdf.fonttype'] = 42 # set font for export to pdfs
matplotlib.rcParams['ps.fonttype'] = 42
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 200
#%%
def load_trajectory_data_cma(Droot, sumdir="summary"):
    subdirs = os.listdir(Droot)
    for unitdir in tqdm(subdirs):
        # 'resnet50_linf8_.Linearfc_009_ns0.5'
        toks = re.findall("(.*)_(\.[^_]*)_(\d*)(.*)", unitdir)
        assert len(toks) == 1, "cannot match pattern for %s" % unitdir
        toks = toks[0]
        netname = toks[0]
        layername = toks[1]
        channum = int(toks[2])
        if len(toks[3]) > 0:
            nstok = re.findall("_ns(.*)", toks[3])[0]
            noise_level = float(nstok)
        else:
            noise_level = 0.0
        npzpaths = glob(join(Droot, unitdir, "CholeskyCMAES_*.npz"))
        for fpath in npzpaths:
            shortfn = os.path.split(fpath)[1]
            # CholeskyCMAES_alexnet_.classifier.Linear6_003_77176.npz
            toks = re.findall("CholeskyCMAES_(.*)_(\.[^_]*)_(\d*)_(\d\d\d\d\d).npz", shortfn)
            assert len(toks) == 1, "cannot match pattern for %s"%shortfn
            assert toks[0][0] == netname
            assert toks[0][1] == layername
            assert channum == int(toks[0][2])
            RND = int(toks[0][3])
            expmeta = EasyDict(netname=netname, layername=layername, channum=channum,
                               noise_level=noise_level, RND=RND,
                               expdir=unitdir)
            data = np.load(fpath)
            # generations = generations, codes_all = codes_all,
            # scores_all = scores_all, cleanscores_all = cleanscores_all, runtime = runtime
            codes_all = data["codes_all"]
            generations = data["generations"]
            scores_all = data["scores_all"]
            cleanscores_all = data["cleanscores_all"]
            # maxobj_data = {k: subd["maxobj"] for k, subd in data.items()}
            # cleanobj_data = {k: subd["cleanscores_all"].max() for k, subd in data.items()}
            # runtime_data = {k: subd["runtime"] for k, subd in data.items()}
            # codenorm_data = {k: subd["codenorm"] for k, subd in data.items()}
            # maxobj_data.update(expmeta)
            # cleanobj_data.update(expmeta)
            # runtime_data.update(expmeta)
            # codenorm_data.update(expmeta)
            # maxobj_col.append(maxobj_data)
            # cleanobj_col.append(cleanobj_data)
            # runtime_col.append(runtime_data)
            # codenorm_col.append(codenorm_data)
            # optimlist = [*data.keys()]
            break
        break
    # maxobj_df = pd.DataFrame(maxobj_col)
    # cleanobj_df = pd.DataFrame(cleanobj_col)
    # runtime_df = pd.DataFrame(runtime_col)
    # codenorm_df = pd.DataFrame(codenorm_col)
    # maxobj_df.to_csv(join(sumdir, "CMA_benchmark_maxobj_summary.csv"))
    # cleanobj_df.to_csv(join(sumdir, "CMA_benchmark_cleanobj_summary.csv"))
    # runtime_df.to_csv(join(sumdir, "CMA_benchmark_runtime_summary.csv"))
    # codenorm_df.to_csv(join(sumdir, "CMA_benchmark_codenorm_summary.csv"))
    return data # maxobj_df, cleanobj_df, runtime_df, codenorm_df, optimlist

dataroot = r"E:\Cluster_Backup\cma_optim_cmp"
data = load_trajectory_data_cma(dataroot, sumdir="summary")
#%%
codes_all = data["codes_all"]
generations = data["generations"]
scores_all = data["scores_all"]
cleanscores_all = data["cleanscores_all"]
meancodes = np.array([codes_all[generations == i,:].mean(axis=0) for i in range(generations.max()+1)])
#%%
PCmachine = PCA(n_components=100).fit(codes_all)
PCcoefs = PCmachine.transform(codes_all)
PCcoefs_mean = PCmachine.transform(meancodes)
#%% All data points
plt.figure()
for i in range(12):
    plt.scatter(generations, PCcoefs[:, i], 9, alpha=0.4)
plt.show()
#%% Mean of each generation
plt.figure()
for i in range(12,20):
    plt.scatter(range(75), PCcoefs_mean[:, i], 9, alpha=0.4)
plt.show()
#%%
expvar_ratio = PCmachine.explained_variance_ratio_
cumexpvar = expvar_ratio.cumsum()
for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
    print(f" {(cumexpvar < r).sum()+1:d} D explaining > {r}")
#%%
from scipy.optimize import curve_fit
def test_func(x, A, omega, phi, bsl):
    y = A*np.cos(omega * (x + phi) * 2 * np.pi) + bsl
    return y

xarr = np.arange(75) / 75
param_col = []
for PCi in range(1, 51):
    curv = PCcoefs_mean[:, PCi-1]
    Rcorr_init = np.corrcoef(test_func(xarr, *[curv.max(), PCi/2, 0, 0]), curv)[0,1]
    print(f"PC{PCi:d} Before correlation {Rcorr_init:.4f}")
    phi_offset = -0.5 if Rcorr_init < -0.8 else 0.0
    params, params_covariance = curve_fit(test_func, xarr, curv,
                                          p0=[curv.max(), PCi/2, phi_offset, 0],
                                          bounds=([0, 0, -1, curv.min()],
                                                   [np.inf, np.inf, 1, curv.max()]),
                                          maxfev=1E4)
    A, omega, phi, bsl = params
    Rcorr = np.corrcoef(test_func(xarr, *params), curv)[0,1]
    print(f"Fitted curve {A:.1f} cos(2pi {omega:.3f}(x + {phi:.3f})) + {bsl:.3f}")
    print(f"PC{PCi:d} Correlation {Rcorr:.4f} ")
    param_col.append(EasyDict(PCi=PCi, A=A, omega=omega, phi=phi, bsl=bsl,
                              Rcorr_init=Rcorr_init, Rcorr=Rcorr))

#%%
cosfit_df = pd.DataFrame(param_col)
cosfit_df["expvar"] = expvar_ratio[:50]
#%%
plt.figure()
plt.plot(cosfit_df.PCi, cosfit_df.A)
plt.show()
plt.figure()
plt.plot(cosfit_df.PCi, cosfit_df.expvar)
plt.show()
plt.figure()
plt.plot(cosfit_df.PCi, cosfit_df.Rcorr)
plt.show()
#%%
plt.figure()
plt.plot(cosfit_df.PCi, cosfit_df.phi)
plt.show()