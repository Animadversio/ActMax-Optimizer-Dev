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
#%%
from scipy.stats import spearmanr, pearsonr, ks_2samp
figdir = r"E:\OneDrive - Harvard University\GECCO2022\Figures\EigenSpaceAlignment"

#%%
G = get_GAN()
#%%
print("mean norm of evolved codes %.1f" % np.linalg.norm(codes_arr, axis=1).mean())
imgs = G.visualize_batch_np(codes_arr)
ToPILImage()(make_grid(imgs)).show()
#%% Measuring the dimensionality of these latent codes.
codes_pool_arr = np.array(sum(code_col.values(), []))
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
np.save(join("data", "codes_pool.npy"), codes_pool_arr)
np.save(join("data", "codes_spect_shuffle.npy"), S_shfl_arr)
#%%
plt.figure()
plt.hist(np.linalg.norm(codes_pool_arr, axis=1),50)
plt.show()
#%%
expvar = S**2 / (S**2).sum()
expvar_shfl = S_shfl_arr**2 / (S_shfl_arr**2).sum(axis=1,keepdims=True)
#%%
signif_axes95 = (expvar > np.percentile(expvar_shfl, 95))
signif_axes99 = (expvar > np.percentile(expvar_shfl, 99))
signif_axes995 = (expvar > np.percentile(expvar_shfl, 99.5))
print("0.05 signif SV dimensions %d"%signif_axes95.sum())
print("0.01 signif SV dimensions %d"%signif_axes99.sum())
print("0.005 signif SV dimensions %d"%signif_axes995.sum())

plt.figure(figsize=(5, 4.5))
plt.plot(expvar_shfl[0, :100], c="k", alpha=0.05, lw=0.2, label="Shuffled")
plt.plot(expvar_shfl[:, :100].T, c="k", alpha=0.05, lw=0.2)
plt.plot(expvar[:100], lw=1.0, label="Evol direction")
plt.xlabel("Singular Value number")
plt.ylabel("Exp. Variance")
plt.title("Total expvar of top 100 dim %.2f%%"%(100 * expvar[:100].sum()))
plt.vlines(signif_axes995.sum(), *plt.ylim(), label="p<0.005 dimensions", ls="-.", color="k", lw=0.5)
plt.vlines(signif_axes95.sum(), *plt.ylim(), label="p<0.05 dimensions", ls="-.", color="k", lw=0.5)
plt.legend()
plt.savefig(join(figdir, "expvar_cmp_with_shfl_trajs.png"))
plt.savefig(join(figdir, "expvar_cmp_with_shfl_trajs.pdf"))
plt.show()
#%%
PC_trajcol = PCA(100).fit(codes_pool_arr@eigvecs[:,:500])
#%%
PC_trajcol_shfl = PCA(100).fit(codes_pool_shuffle@eigvecs[:,:500])
#%%
plt.figure()
plt.plot(PC_trajcol.explained_variance_ratio_)
plt.plot(PC_trajcol_shfl.explained_variance_ratio_)
plt.show()
#%%
signif_axes95 = (expvar > np.percentile(expvar_shfl, 95))
signif_axes99 = (expvar > np.percentile(expvar_shfl, 99))
signif_axes995 = (expvar > np.percentile(expvar_shfl, 99.5))
print("0.05 signif SV dimensions %d"%signif_axes95.sum())
print("0.01 signif SV dimensions %d"%signif_axes99.sum())
print("0.005 signif SV dimensions %d"%signif_axes995.sum())


#%% Load up the Hessian Eigenframe
Hdata = np.load(r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\fc6GAN\Evolution_Avg_Hess.npz")
eigvecs = Hdata["eigvect_avg"].copy()[:, ::-1]
eigvals = Hdata["eigv_avg"].copy()[::-1]
eigids = np.arange(1, 4097)
#%%
projcoef = codes_pool_arr@eigvecs
projcoef_shfl = codes_pool_shuffle@eigvecs

#%%
cutoff = 800
meanabs_coef = np.abs(projcoef).mean(axis=0)
meanabs_coef_shfl = np.abs(projcoef_shfl).mean(axis=0)
plt.figure(figsize=(5,5.5))
plt.scatter(np.arange(1, 4097), meanabs_coef, s=16,  alpha=0.15, label="Evol Direction")
plt.scatter(np.arange(1, 4097), meanabs_coef_shfl, s=16, alpha=0.15, label="Shuffled")
plt.vlines(cutoff, *plt.ylim(),ls="-.",color="black", label="eig cutoff")
plt.xlim([-5, 4100])
rval_id, pval_id = spearmanr(meanabs_coef[:cutoff], eigids[:cutoff])
ksstats_pre, kspval_pre = ks_2samp(meanabs_coef[:cutoff], meanabs_coef_shfl[:cutoff])
ksstats_pst, kspval_pst = ks_2samp(meanabs_coef[cutoff:], meanabs_coef_shfl[cutoff:])
print(f"Corr with top {cutoff} eigenvalues")
print(f"Spearman corr = {rval_id:.3f}, {pval_id:.1e}")
print(f"KS test of top {cutoff} amplitudes {ksstats_pre:.3f}, {kspval_pre:.1e}")
print(f"KS test of beyond top {cutoff} amplitudes {ksstats_pst:.3f}, {kspval_pst:.1e}")
plt.xlabel("PC number")
plt.ylabel("Mean proj amplitude \\bar{|z_Tv_k^T|}")
plt.title(f"Spearman corr with top {cutoff} eigenvalue rank \n corr = {rval_id:.3f} (P={pval_id:.1e})\n"
          f"top {cutoff}, KS test={ksstats_pre:.3f}, {kspval_pre:.1e}\n"
          f"beyond top {cutoff}, KS test={ksstats_pst:.3f}, {kspval_pst:.1e}")
plt.legend()
plt.savefig(join(figdir, "meanabs_coef_PC_dist_stats.png"))
plt.savefig(join(figdir, "meanabs_coef_PC_dist_stats.pdf"))
plt.show()

#%%
cutoff = 800
absmean_coef = np.abs(projcoef.mean(axis=0))
absmean_coef_shfl = np.abs(projcoef_shfl.mean(axis=0))
plt.figure(figsize=(5, 5.5))
plt.scatter(np.arange(1, 4097), absmean_coef, s=16,  alpha=0.15, label="Evol Direction")
plt.scatter(np.arange(1, 4097), absmean_coef_shfl, s=16, alpha=0.15, label="Shuffled")
plt.vlines(cutoff, *plt.ylim(), ls="-.", color="black", label="eig cutoff")
plt.xlim([-5, 4100])
plt.ylim([-0.1, 1.3])
rval_id, pval_id = spearmanr(absmean_coef[:cutoff], eigids[:cutoff])
ksstats_pre, kspval_pre = ks_2samp(absmean_coef[:cutoff], absmean_coef_shfl[:cutoff])
ksstats_pst, kspval_pst = ks_2samp(absmean_coef[cutoff:], absmean_coef_shfl[cutoff:])
print(f"Spearman Corr with top {cutoff} eigenvalues order = {rval_id:.3f}, {pval_id:.1e}")
print(f"KS test of top {cutoff} amplitudes {ksstats_pre:.3f}, {kspval_pre:.1e}")
print(f"KS test of beyond top {cutoff} amplitudes {ksstats_pst:.3f}, {kspval_pst:.1e}")
plt.xlabel("PC number")
plt.ylabel("Amplitude of Mean proj coef |\\bar{z_Tv_k^T}|")
plt.title(f"Spearman corr with top {cutoff} eigenvalue rank \n corr = {rval_id:.3f} (P={pval_id:.1e})\n"
          f"top {cutoff}, KS test={ksstats_pre:.3f}, {kspval_pre:.1e}\n"
          f"beyond top {cutoff}, KS test={ksstats_pst:.3f}, {kspval_pst:.1e}")
plt.legend()
plt.savefig(join(figdir, "absmean_coef_PC_dist_stats.png"))
plt.savefig(join(figdir, "absmean_coef_PC_dist_stats.pdf"))
plt.show()
#%%
plt.figure(figsize=[5.5, 5])
plt.scatter(np.log10(eigvals), np.abs(projcoef.mean(axis=0)), s=9, alpha=0.15, label="Evol Direction")
plt.scatter(np.log10(eigvals), np.abs(projcoef_shfl.mean(axis=0)), s=9,  alpha=0.15, label="Shuffled")
plt.xlabel("log 10 (eig value)")
plt.ylabel("Amplitude of Mean proj coef |\\bar{z_Tv_k^T}|")
plt.legend()
plt.savefig(join(figdir, "absmean_coef_scatter.png"))
plt.savefig(join(figdir, "absmean_coef_scatter.pdf"))
plt.show()
#%%
plt.figure()
plt.scatter(np.log10(eigvals), np.abs(projcoef).mean(axis=0), s=9, alpha=0.15, label="Evol Direction")
plt.scatter(np.log10(eigvals), np.abs(projcoef_shfl).mean(axis=0), s=9,  alpha=0.15, label="Shuffled")
plt.xlabel("log 10 (eig value)")
plt.ylabel("Mean Proj Amplitude |\\bar{z_Tv_k^T}|")
plt.legend()
plt.savefig(join(figdir, "meanabs_coef_scatter.png"))
plt.savefig(join(figdir, "meanabs_coef_scatter.pdf"))
plt.show()

#%%
spearmanr(np.abs(projcoef_shfl).mean(axis=0), np.arange(1, 4097))
#%%

#%%
cutoff = 1000
rval_id, pval_id = spearmanr(np.abs(projcoef).mean(axis=0)[:cutoff], np.arange(1, cutoff+1))
rval_eig, pval_eig = spearmanr(np.abs(projcoef).mean(axis=0)[:cutoff], eigvals[:-cutoff-1:-1])
rval_logeig, pval_logeig = spearmanr(np.abs(projcoef).mean(axis=0)[:cutoff], np.log10(eigvals[:-cutoff-1:-1]), nan_policy="omit")
print(f"With cutoff at eigenvalue {cutoff}")
print(f"corr = {rval_id:.3f}, {pval_id:.1e}")
print(f"corr = {rval_eig:.3f}, {pval_eig:.1e}")
print(f"corr = {rval_logeig:.3f}, {pval_logeig:.1e}")