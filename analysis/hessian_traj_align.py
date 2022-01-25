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
from core.montage_utils import make_grid, ToPILImage
from core.GAN_utils import upconvGAN

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 200
#%%
def get_GAN():
    G = upconvGAN("fc6")
    G.requires_grad_(False)
    G.cuda().eval()
    return G


def load_trajectory_PCAdata_cma(Droot, sumdir="summary"):
    subdirs = os.listdir(Droot)
    df_col = {}
    meta_col = {}
    PCreduc_code_col = {}
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
        csvpaths = glob(join(Droot, unitdir, "CholeskyCMAES_meanPCA_cosfit_*.csv"))
        df_col[unitdir] = []
        meta_col[unitdir] = []
        PCreduc_code_col[unitdir] = []
        for fpath in csvpaths:
            shortfn = os.path.split(fpath)[1]
            # CholeskyCMAES_alexnet_.classifier.Linear6_003_77176.npz
            toks = re.findall("CholeskyCMAES_meanPCA_cosfit_(.*)_(\.[^_]*)_(\d*)_(\d\d\d\d\d).csv", shortfn)
            assert len(toks) == 1, "cannot match pattern for %s"%shortfn
            assert toks[0][0] == netname
            assert toks[0][1] == layername
            assert channum == int(toks[0][2])
            RND = int(toks[0][3])
            npzpath = "CholeskyCMAES_meanPCA_coef_%s_%s_%03d_%05d.npz"%(netname,layername,channum,RND)
            expmeta = EasyDict(netname=netname, layername=layername, channum=channum,
                               noise_level=noise_level, RND=RND,
                               expdir=unitdir)
            PCA_df = pd.read_csv(fpath)
            df_col[unitdir].append(PCA_df)
            meta_col[unitdir].append(expmeta)
            data = np.load(join(Droot,unitdir,npzpath))
            # print(list(data))
            PCreduc_code = data["PCcoefs_mean"][-1, :5] @ data["PCvecs"][:5, :]
            PCreduc_code_col[unitdir].append(PCreduc_code)
            # break
        # break
    return df_col, meta_col, PCreduc_code_col

dataroot = r"E:\Cluster_Backup\cma_optim_cmp"
meandf_col, meta_col, code_col = load_trajectory_PCAdata_cma(dataroot)
#%%
codes_arr = np.array(code_col["alexnet_.classifier.Linear6_000"])
#%%
G = get_GAN()
#%%
print("mean norm of evolved codes %.1f"%np.linalg.norm(codes_arr,axis=1).mean())
imgs = G.visualize_batch_np(codes_arr)
ToPILImage()(make_grid(imgs)).show()

#%% Measuring the dimensionality of these latent codes.
codes_pool_arr = np.array(sum(code_col.values(),[]))
#%%
U, S, V = np.linalg.svd(codes_pool_arr)
#%%
S_shfl_col = []
for i in tqdm(range(500)):
    codes_pool_shuffle = np.array([row[np.random.permutation(4096)] for row in codes_pool_arr])
    U_shfl, S_shfl, V_shfl = np.linalg.svd(codes_pool_shuffle)
    S_shfl_col.append(S_shfl)
#%%
S_shfl_arr = np.array(S_shfl_col)
#%%
np.save(join("data","codes_pool.npy"), codes_pool_arr)
np.save(join("data","codes_spect_shuffle.npy"), S_shfl_arr)
#%%
expvar = S**2 / (S**2).sum()
expvar_shfl = S_shfl_arr**2 / (S_shfl_arr**2).sum(axis=1,keepdims=True)
#%%
plt.figure()
plt.plot(expvar[:100])
plt.plot(expvar_shfl[:,:100].T, c="k", alpha=0.05, lw=0.2)
plt.show()
#%%
signif_axes95 = (expvar > np.percentile(expvar_shfl, 95))
signif_axes99 = (expvar > np.percentile(expvar_shfl, 99))
signif_axes995 = (expvar > np.percentile(expvar_shfl, 99.5))
print("0.05 signif SV dimensions %d"%signif_axes95.sum())
print("0.01 signif SV dimensions %d"%signif_axes99.sum())
print("0.005 signif SV dimensions %d"%signif_axes995.sum())
#%%
Hdata = np.load(r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\fc6GAN\Evolution_Avg_Hess.npz")
#%%
eigvecs = Hdata["eigvect_avg"]
eigvals = Hdata["eigv_avg"]
#%%
projcoef = codes_pool_arr@eigvecs[:,::-1]
projcoef_shfl = codes_pool_shuffle@eigvecs[:,::-1]
#%%
plt.figure()
plt.scatter(np.arange(1, 4097), np.abs(projcoef).mean(axis=0),alpha=0.2)
plt.scatter(np.arange(1, 4097), np.abs(projcoef_shfl).mean(axis=0),alpha=0.2)
plt.show()
#%%
from scipy.stats import spearmanr, pearsonr
spearmanr(np.abs(projcoef_shfl).mean(axis=0), np.arange(1, 4097))
#%%
rval_id, pval_id = spearmanr(np.abs(projcoef).mean(axis=0), np.arange(1, 4097))
rval_eig, pval_eig = spearmanr(np.abs(projcoef).mean(axis=0), eigvals[::-1])
rval_logeig, pval_logeig = spearmanr(np.abs(projcoef).mean(axis=0), np.log10(eigvals[::-1]), nan_policy="omit")
print(f"corr = {rval_id:.3f}, {pval_id:.1e}")
print(f"corr = {rval_eig:.3f}, {pval_eig:.1e}")
print(f"corr = {rval_logeig:.3f}, {pval_logeig:.1e}")
#%%
cutoff = 1000
rval_id, pval_id = spearmanr(np.abs(projcoef).mean(axis=0)[:cutoff], np.arange(1, cutoff+1))
rval_eig, pval_eig = spearmanr(np.abs(projcoef).mean(axis=0)[:cutoff], eigvals[:-cutoff-1:-1])
rval_logeig, pval_logeig = spearmanr(np.abs(projcoef).mean(axis=0)[:cutoff], np.log10(eigvals[:-cutoff-1:-1]), nan_policy="omit")
print(f"corr = {rval_id:.3f}, {pval_id:.1e}")
print(f"corr = {rval_eig:.3f}, {pval_eig:.1e}")
print(f"corr = {rval_logeig:.3f}, {pval_logeig:.1e}")