"""Script for loading the experimental result from Cluster and analyze / visualize the result. """
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
matplotlib.rcParams['pdf.fonttype'] = 42 # set font for export to pdfs
matplotlib.rcParams['ps.fonttype'] = 42
#%%
def load_format_data(Droot, sumdir="summary"):
    maxobj_col = []
    runtime_col = []
    codenorm_col = []
    subdirs = os.listdir(Droot)
    for unitdir in tqdm(subdirs):
        pklpaths = glob(join(Droot, unitdir, "*.pkl"))
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
        for fpath in pklpaths:
            shortfn = os.path.split(fpath)[1]
            # "summarize_alexnet_.classifier.Linear6_001_12514.pkl"
            toks = re.findall("summarize_(.*)_(\.[^_]*)_(\d*)_(\d\d\d\d\d).pkl", shortfn)
            assert len(toks) == 1, "cannot match pattern for %s"%shortfn
            assert toks[0][0] == netname
            assert toks[0][1] == layername
            assert channum == int(toks[0][2])
            RND = int(toks[0][3])
            expmeta = EasyDict(netname=netname, layername=layername, channum=channum,
                               noise_level=noise_level, RND=RND,
                               expdir=unitdir)
            data = pkl.load(open(fpath, "rb"))
            maxobj_data = {k: subd["maxobj"] for k, subd in data.items()}
            runtime_data = {k: subd["runtime"] for k, subd in data.items()}
            codenorm_data = {k: subd["codenorm"] for k, subd in data.items()}
            maxobj_data.update(expmeta)
            runtime_data.update(expmeta)
            codenorm_data.update(expmeta)
            maxobj_col.append(maxobj_data)
            runtime_col.append(runtime_data)
            codenorm_col.append(codenorm_data)
            optimlist = [*data.keys()]

    maxobj_df = pd.DataFrame(maxobj_col)
    runtime_df = pd.DataFrame(runtime_col)
    codenorm_df = pd.DataFrame(codenorm_col)
    maxobj_df.to_csv(join(sumdir, "ng_benchmark_maxobj_summary.csv"))
    runtime_df.to_csv(join(sumdir, "ng_benchmark_runtime_summary.csv"))
    codenorm_df.to_csv(join(sumdir, "ng_benchmark_codenorm_summary.csv"))
    return maxobj_df, runtime_df, codenorm_df, optimlist

dataroot = r"E:\Cluster_Backup\ng_optim_cmp"
maxobj_df, runtime_df, codenorm_df, optimlist = load_format_data(dataroot, sumdir="summary")
#%%
Anet_msk = maxobj_df.netname=="alexnet"
Rnet_msk = maxobj_df.netname=="resnet50_linf8"
clean_msk = maxobj_df.noise_level==0.0
ns02_msk = maxobj_df.noise_level==0.0
ns05_msk = maxobj_df.noise_level==0.5
#%%

plt.figure(figsize=[8, 6])
normmaxobj_df.boxplot(column=optimlist)
plt.show()
#%%
normmaxobj_df = maxobj_df[optimlist].\
    divide(maxobj_df[optimlist].mean(axis=1), axis=0)
plt.figure(figsize=[8,6])
normmaxobj_df.boxplot(column=optimlist)
plt.show()
#%%
figdir = r"H:\GECCO2022\Figures\NGBenchmark"
optimorder = ['CMA','DiagonalCMA','SQPCMA','RescaledCMA','ES',
 'NGOpt','DE','TwoPointsDE',
 'PSO','OnePlusOne','TBPSA',
 'RandomSearch']
#%%
normalizer = "mean"
if normalizer == "mean":
    normmaxobj_df = maxobj_df[optimlist].\
        divide(maxobj_df[optimlist].mean(axis=1), axis=0)
elif normalizer == "max":
    normmaxobj_df = maxobj_df[optimlist].\
        divide(maxobj_df[optimlist].max(axis=1), axis=0)
elif normalizer == "RSmean":
    normmaxobj_df = maxobj_df[optimlist].\
        divide(maxobj_df["RandomSearch"], axis=0)
else:
    raise NotImplementedError

#%%
msklist = [Anet_msk, Rnet_msk, clean_msk, ns02_msk, ns05_msk,
           Anet_msk & clean_msk, Anet_msk & ns02_msk, Anet_msk & ns05_msk,
           Rnet_msk & clean_msk, Rnet_msk & ns02_msk, Rnet_msk & ns05_msk]
mskstrlist = ["alexnet", "resnet-robust",
              "noise-free", "noise0.2", "noise0.5",
              "alexnet_noise-free", "alexnet_noise0.2", "alexnet_noise0.5",
              "resnet_noise-free", "resnet_noise0.2", "resnet_noise0.5"]
for msk, msklabel in zip(msklist, mskstrlist):
    plt.figure(figsize=[8, 6])
    sns.boxplot(data=normmaxobj_df[msk], order=optimorder, width=0.7)
    # sns.violinplot(data=normmaxobj_df[clean_msk], order=optimorder, width=1.1)
    plt.xticks(rotation=25)
    plt.ylabel("activation (normalized by %s)"%normalizer)
    plt.title("Optimizer Comparsion for %s"%msklabel)
    plt.savefig(join(figdir, "optim_cmp_benchmark_%s_%s.png"%(normalizer, msklabel)))
    plt.savefig(join(figdir, "optim_cmp_benchmark_%s_%s.pdf"%(normalizer, msklabel)))
    plt.show()
#%%
for msk, msklabel in zip(msklist[:2], mskstrlist[:2]):
    plt.figure(figsize=[8, 6])
    sns.boxplot(data=runtime_df[msk], order=optimorder, width=0.7)
    # sns.violinplot(data=normmaxobj_df[clean_msk], order=optimorder, width=1.1)
    plt.xticks(rotation=25)
    plt.ylabel("runtime (sec)")
    plt.title("Optimizer Comparsion for %s"%msklabel)
    plt.savefig(join(figdir, "optim_runtime_cmp_benchmark_%s.png"%(msklabel)))
    plt.savefig(join(figdir, "optim_runtime_cmp_benchmark_%s.pdf"%(msklabel)))
    plt.show()
#%%
runtime_df[Anet_msk][optimorder].mean(axis=0)
runtime_df[Anet_msk][optimorder].describe()
#%%
# note these dataframes are wide format table
plt.figure()
maxobj_df.boxplot(column=optimlist)
plt.show()

#%%

# def scan_hess_npz(Hdir, npzpat="Hess_BP_(\d*).npz", evakey='eva_BP', evckey='evc_BP', featkey=None):
#     """ Function to load in npz and collect the spectra.
#     Set evckey=None to avoid loading eigenvectors.

#     Note for newer experiments use evakey='eva_BP', evckey='evc_BP'
#     For older experiments use evakey='eigvals', evckey='eigvects'"""
#     npzpaths = glob(join(Hdir, "*.npz"))
#     npzfns = [path.split("\\")[-1] for path in npzpaths]
#     npzpattern = re.compile(npzpat)
#     eigval_col = []
#     eigvec_col = []
#     feat_col = []
#     meta = []
#     for fn, path in tqdm(zip(npzfns, npzpaths)):
#         match = npzpattern.findall(fn)
#         if len(match) == 0:
#             continue
#         parts = match[0]  # trunc, RND
#         data = np.load(path)
#         try:
#             evas = data[evakey]
#             eigval_col.append(evas)
#             if evckey is not None:
#                 evcs = data[evckey]
#                 eigvec_col.append(evcs)
#             if featkey is not None:
#                 feat = data[featkey]
#                 feat_col.append(feat)
#             meta.append(parts)
#         except KeyError:
#             print("KeyError, keys in the archive : ", list(data))
#             return
#     eigval_col = np.array(eigval_col)
#     print("Load %d npz files of Hessian info" % len(meta))
#     if featkey is None:
#         return eigval_col, eigvec_col, meta
#     else:
#         feat_col = np.array(tuple(feat_col)).squeeze()
#         return eigval_col, eigvec_col, feat_col, meta

# # Develop zone
# sumdir = "summary"
# Droot = dataroot
# maxobj_col = []
# runtime_col = []
# codenorm_col = []
# subdirs = os.listdir(Droot)
# for unitdir in tqdm(subdirs):
#     pklpaths = glob(join(Droot, unitdir, "*.pkl"))
#     toks = re.findall("(.*)_(\.[^_]*)_(\d*)(.*)", unitdir)[0]
#     netname = toks[0]
#     layername = toks[1]
#     channum = int(toks[2])
#     if len(toks[3]) > 0:
#         nstok = re.findall("_ns(.*)", toks[3])[0]
#         noise_level = float(nstok[0])
#     else:
#         noise_level = 0.0
#     for fpath in pklpaths:
#         shortfn = os.path.split(fpath)[1]
#         # "summarize_alexnet_.classifier.Linear6_001_12514.pkl"
#         toks = re.findall("summarize_(.*)_(\.[^_]*)_(\d*)_(\d\d\d\d\d).pkl", shortfn)
#         assert len(toks) == 1
#         assert toks[0][0] == netname
#         assert toks[0][1] == layername
#         assert channum == int(toks[0][2])
#         RND = int(toks[0][3])
#         expmeta = EasyDict(netname=netname, layername=layername, channum=channum, noise_level=noise_level, RND=RND, expdir=unitdir)
#         data = pkl.load(open(fpath, "rb"))
#         maxobj_data = {k: subd["maxobj"] for k, subd in data.items()}
#         runtime_data = {k: subd["runtime"] for k, subd in data.items()}
#         codenorm_data = {k: subd["codenorm"] for k, subd in data.items()}
#         maxobj_data.update(expmeta)
#         runtime_data.update(expmeta)
#         codenorm_data.update(expmeta)
#         maxobj_col.append(maxobj_data)
#         runtime_col.append(runtime_data)
#         codenorm_col.append(codenorm_data)

# maxobj_df = pd.DataFrame(maxobj_col)
# runtime_df = pd.DataFrame(runtime_col)
# codenorm_df = pd.DataFrame(codenorm_col)
# #%%
# maxobj_df.to_csv(join(sumdir, "ng_benchmark_maxobj_summary.csv"))
# runtime_df.to_csv(join(sumdir, "ng_benchmark_runtime_summary.csv"))
# codenorm_df.to_csv(join(sumdir, "ng_benchmark_codenorm_summary.csv"))
