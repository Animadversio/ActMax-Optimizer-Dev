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
from scipy.stats import spearmanr, pearsonr, ks_2samp
figdir = r"E:\OneDrive - Harvard University\GECCO2022\Figures\EigenSpaceAlignment"

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
def load_finalgen_cma_noisectrl(Droot, select_optim=None, PCproj=None):
    codes_col = []
    meta_col = []
    subdirs = os.listdir(Droot)
    if select_optim is not None:
        subdirs = select_optim
    for optimnm in tqdm(subdirs):
        npzpaths = glob(join(Droot, optimnm, "noisectrl_%s_*.npz"%optimnm))
        for fpath in tqdm(npzpaths):
            if "PCA" in fpath:
                continue
            shortfn = os.path.split(fpath)[1]
            # CholeskyCMAES_alexnet_.classifier.Linear6_003_77176.npz
            try:
                toks = re.findall("noisectrl_%s_rep(\d\d\d\d\d).npz"%optimnm, shortfn)
                RND = int(toks[0][0])
                expmeta = EasyDict(RND=RND, expdir=optimnm)
                data = np.load(fpath)
                codes_all = data["codes_all"]
                generations = data["generations"]
                scores_all = data["scores_all"]
                meancodes = np.array([codes_all[generations == i, :].mean(axis=0)
                                      for i in range(generations.max() + 1)])
                evoldir = meancodes[-1,:]
                if PCproj is not None:
                    PCA_npz = "%s_meanPCA_coef_rep%05d.npz"%(optimnm,RND)
                    PCA_data = np.load(join(Droot, optimnm, PCA_npz))
                    evoldir = meancodes[-1,:] @ PCA_data["PCvecs"][:5, :].T @PCA_data["PCvecs"][:5, :]#data["PCcoefs_mean"][-1, :5] @ data["PCvecs"][:5, :]
                meta_col.append(expmeta)
                codes_col.append(evoldir)
            except :
                continue
    meta_df = pd.DataFrame(meta_col)
    return codes_col, meta_df

dataroot = r"F:\insilico_exps\noise_optim_ctrl"
ctrlcodes_col, meta_df = load_finalgen_cma_noisectrl(dataroot, select_optim=["CholeskyCMAES"])
ctrlcodes_arr = np.array(ctrlcodes_col)
#%%
def load_finalgen_cma(Droot, sumdir="summary"):
    subdirs = os.listdir(Droot)
    # df_col = {}
    meta_col = {}
    fincode_col = {}
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
        # df_col[unitdir] = []
        meta_col[unitdir] = []
        fincode_col[unitdir] = []
        for fpath in csvpaths:
            shortfn = os.path.split(fpath)[1]
            # CholeskyCMAES_alexnet_.classifier.Linear6_003_77176.npz
            toks = re.findall("CholeskyCMAES_meanPCA_cosfit_(.*)_(\.[^_]*)_(\d*)_(\d\d\d\d\d).csv", shortfn)
            assert len(toks) == 1, "cannot match pattern for %s"%shortfn
            assert toks[0][0] == netname
            assert toks[0][1] == layername
            assert channum == int(toks[0][2])
            RND = int(toks[0][3])
            # npzpath = "CholeskyCMAES_meanPCA_coef_%s_%s_%03d_%05d.npz"%(netname,layername,channum,RND)
            npzpath = "CholeskyCMAES_%s_%s_%03d_%05d.npz"%(netname,layername,channum,RND)
            expmeta = EasyDict(netname=netname, layername=layername, channum=channum,
                               noise_level=noise_level, RND=RND,
                               expdir=unitdir)
            # PCA_df = pd.read_csv(fpath)
            # df_col[unitdir].append(PCA_df)# CholeskyCMAES_alexnet_.classifier.Linear6_000_89223.npz

            meta_col[unitdir].append(expmeta)
            data = np.load(join(Droot,unitdir,npzpath))
            codes_all = data["codes_all"]
            generations = data["generations"]
            scores_all = data["scores_all"]
            meancodes = np.array([codes_all[generations == i, :].mean(axis=0)
                                  for i in range(generations.max() + 1)])
            evoldir = meancodes[-1, :]
            # print(list(data))
            # PCreduc_code = data["PCcoefs_mean"][-1, :5] @ data["PCvecs"][:5, :]
            fincode_col[unitdir].append(evoldir)
            # break
        # break
    return fincode_col, meta_col

dataroot = r"E:\Cluster_Backup\cma_optim_cmp"
fincode_col, meta_col = load_finalgen_cma(dataroot)
#%%%
fincodes_arr = np.array(sum([v for k, v in fincode_col.items()],[]))
#%%
np.savez("data\\finalcodes_evol_vs_ctrl.npz", fincodes_arr=fincodes_arr,
         meta_col=meta_col, ctrlcodes_arr=ctrlcodes_arr)

#%%
codes_arr = np.array(code_col["alexnet_.classifier.Linear6_000"])
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
U, S, V = np.linalg.svd(fincodes_arr)
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
#%% Second Major analysis

codes_pool_arr = np.load(join("data", "codes_pool.npy"), ).copy()
codes_pool_shuffle = np.array([row[np.random.permutation(4096)] for row in codes_pool_arr])
#%%
data = np.load("data\\finalcodes_evol_vs_ctrl.npz")
fincodes_arr = data["fincodes_arr"]
ctrlcodes_arr = data["ctrlcodes_arr"]

fincodes_shfl = np.array([row[np.random.permutation(4096)] for row in fincodes_arr])
#%% Load up the Hessian Eigenframe
Hdata = np.load(r"E:\OneDrive - Washington University in St. Louis\Hessian_summary\fc6GAN\Evolution_Avg_Hess.npz")
eigvecs = np.array(Hdata["eigvect_avg"].copy()[:, ::-1])
eigvals = np.array(Hdata["eigv_avg"].copy()[::-1])
eigids = np.arange(1, 4097)
#%%
projcoef = codes_pool_arr@eigvecs
projcoef_shfl = codes_pool_shuffle@eigvecs
#%%
realevol_projcoef = fincodes_arr @ eigvecs
evolshfl_projcoef = fincodes_shfl @ eigvecs
ctrlevol_projcoef = ctrlcodes_arr @eigvecs
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
plt.figure(figsize=[4.5, 4])
plt.scatter(np.log10(eigvals), np.abs(projcoef).mean(axis=0), s=6, alpha=0.15, label="Evol Direction")
plt.scatter(np.log10(eigvals), np.abs(projcoef_shfl).mean(axis=0), s=6,  alpha=0.15, label="Shuffled")
plt.scatter(np.log10(eigvals), np.abs(ctrlevol_projcoef).mean(axis=0), s=6, alpha=0.05, label="ctrl Evol Direction")
plt.xlabel("log 10 (eig value)")
plt.ylabel("Mean Proj Amplitude |\\bar{z_Tv_k^T}|")
plt.legend()
# plt.savefig(join(figdir, "meanabs_coef_scatter.pdf"))
# plt.savefig(join(figdir, "meanabs_coef_scatter.png"))
plt.show()
#%%
cutoff = 800
ks_prepost_nois, kspval_prepost_nois = ks_2samp(ctrlevol_projcoef[:,:cutoff].flatten(), ctrlevol_projcoef[:,cutoff:].flatten())
ks_prepost_shfl, kspval_prepost_shfl = ks_2samp(evolshfl_projcoef[:,:cutoff].flatten(), evolshfl_projcoef[:,cutoff:].flatten())
ks_prepost_real, kspval_prepost_real = ks_2samp(realevol_projcoef[:,:cutoff].flatten(), realevol_projcoef[:,cutoff:].flatten())
#%%
rval_log_real, pval_log_real = pearsonr(np.abs(realevol_projcoef).mean(axis=0)[:cutoff], np.log10(eigvals[:cutoff]), )
rval_log_evol, pval_log_evol = pearsonr(np.abs(evolshfl_projcoef).mean(axis=0)[:cutoff], np.log10(eigvals[:cutoff]), )
rval_log_ctrl, pval_log_ctrl = pearsonr(np.abs(ctrlevol_projcoef).mean(axis=0)[:cutoff], np.log10(eigvals[:cutoff]), )

#%%
spearmanr(np.abs(projcoef_shfl).mean(axis=0), np.arange(1, 4097))
#%%

def add_reg_line(x,y, ax=None, label=""):
    m, b = np.polyfit(x, y, 1)
    #m = slope, b=intercept
    xbnd = np.array([x.min(), x.max()])
    plt.plot(xbnd, m*xbnd + b, color="r", label=f"{label} fit: y={m:.1e}x+{b:.1e}")

#%% Final version of

plt.figure(figsize=[4, 4.5])
plt.scatter(np.log10(eigvals), np.abs(ctrlevol_projcoef).mean(axis=0), s=6, alpha=0.15, label="Ctrl Noise Evol N=100")
plt.scatter(np.log10(eigvals), np.abs(evolshfl_projcoef).mean(axis=0), s=6,  alpha=0.15, label="Shuffled N=1050")
plt.scatter(np.log10(eigvals), np.abs(realevol_projcoef).mean(axis=0), s=6, alpha=0.15, label="Evol Direction N=1050")
plt.vlines(np.log10(eigvals)[800], *plt.ylim(),linestyle="-.",color='k')
add_reg_line(np.log10(eigvals)[:800],np.abs(ctrlevol_projcoef).mean(axis=0)[:800],label="ctrl")
add_reg_line(np.log10(eigvals)[:800],np.abs(evolshfl_projcoef).mean(axis=0)[:800],label="shfl")
add_reg_line(np.log10(eigvals)[:800],np.abs(realevol_projcoef).mean(axis=0)[:800],label="evol")
plt.xlabel("log 10 (eig value)")
plt.ylabel("Mean Travel distance \\bar{|z_Tv_k^T|}")
plt.legend()
plt.title(f"KS test of coefficient distribution\n pre and post cutoff {cutoff}\n"
          f"KS = {ks_prepost_real:.3f} p = {kspval_prepost_real:.1e}\n"
          f"Pearson correlation between top {cutoff}\n amplitude and the log eigenvalue\n"
          f"Rho = {rval_log_real:.3f} p= {pval_log_real:.1e}")
plt.savefig(join(figdir, "meanabs_coef_scatter_new_noise_ctrl_with_regress.pdf"))
plt.savefig(join(figdir, "meanabs_coef_scatter_new_noise_ctrl_with_regress.png"))
plt.show()

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
#%%
ks_prepost_real_mean, kspval_prepost_real_mean = ks_2samp(np.mean(realevol_projcoef[:,:cutoff],axis=0),
                                                          np.mean(realevol_projcoef[:,cutoff:],axis=0))
#%%