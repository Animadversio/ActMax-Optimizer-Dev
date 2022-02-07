"""Analysis script for understanding the
PCA spectral structure of Evolution trajectories
"""
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
#%%
def load_trajectory_data_cma(Droot, sumdir="summary", meanPCA=True, fullPCA=False):
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
            if "PCA" in fpath:
                continue
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
            meancodes = np.array([codes_all[generations == i, :].mean(axis=0) for i in range(generations.max() + 1)])
            # if fullPCA:
            #     PCmachine = PCA(n_components=50).fit(codes_all)
            #     PCcoefs = PCmachine.transform(codes_all)
            #     PCcoefs_mean = PCmachine.transform(meancodes)
            #     expvar_ratio = PCmachine.explained_variance_ratio_
            #     cumexpvar = expvar_ratio.cumsum()
            #     np.savez(join(Droot, unitdir, "CholeskyCMAES_PCA_coef_%s_%s_%03d_%05d.npz"\
            #                           %(netname, layername, channum, RND)),
            #             PCcoefs=PCcoefs, PCcoefs_mean=PCcoefs_mean, PCvecs=PCmachine.components_,
            #             expvar_ratio=expvar_ratio, cumexpvar=cumexpvar
            #             )
            #     xarr = np.arange(75) / 75
            #     param_col = []
            #     for PCi in range(1, 51):
            #         curv = PCcoefs_mean[:, PCi - 1]
            #         Rcorr_init = np.corrcoef(test_func(xarr, *[curv.max(), PCi / 2, 0, 0]), curv)[0, 1]
            #         print(f"PC{PCi:d} Before correlation {Rcorr_init:.4f}")
            #         phi_offset = -0.5 if Rcorr_init < -0.8 else 0.0
            #         try:
            #             params, params_covariance = curve_fit(test_func, xarr, curv,
            #                                                   p0=[curv.max(), PCi / 2, phi_offset, 0],
            #                                                   bounds=([0, 0, -1, curv.min()],
            #                                                           [np.inf, np.inf, 1, curv.max()]),
            #                                                   maxfev=1E4)
            #             A, omega, phi, bsl = params
            #             Rcorr = np.corrcoef(test_func(xarr, *params), curv)[0, 1]
            #             print(f"Fitted curve {A:.1f} cos(2pi {omega:.3f}(x + {phi:.3f})) + {bsl:.3f}")
            #             print(f"PC{PCi:d} Correlation {Rcorr:.4f} ")
            #         except RuntimeError:
            #             A, omega, phi, bsl = np.nan, np.nan, np.nan, np.nan
            #             Rcorr = 0
            #         param_col.append(EasyDict(PCi=PCi, A=A, omega=omega, phi=phi, bsl=bsl,
            #                                   Rcorr_init=Rcorr_init, Rcorr=Rcorr))
            #
            #     cosfit_df = pd.DataFrame(param_col)
            #     cosfit_df["expvar"] = expvar_ratio[:50]
            #     cosfit_df.to_csv(join(Droot, unitdir, "CholeskyCMAES_PCA_cosfit_%s_%s_%03d_%05d.csv"\
            #                           %(netname, layername, channum, RND)))
            if meanPCA:
                PCmachine_m = PCA(n_components=75).fit(meancodes)
                PCcoefs_mean = PCmachine_m.transform(meancodes)
                expvar_ratio = PCmachine_m.explained_variance_ratio_
                cumexpvar = expvar_ratio.cumsum()
                np.savez(join(Droot, unitdir, "CholeskyCMAES_meanPCA_coef_%s_%s_%03d_%05d.npz" \
                              % (netname, layername, channum, RND)),
                         PCcoefs_mean=PCcoefs_mean, PCvecs=PCmachine_m.components_,
                         expvar_ratio=expvar_ratio, cumexpvar=cumexpvar
                         )
                xarr = np.arange(75) / 75
                param_col = []
                for PCi in range(1, 75):
                    curv = PCcoefs_mean[:, PCi - 1]
                    Rcorr_init = np.corrcoef(test_func(xarr, *[curv.max(), PCi / 2, 0, 0]), curv)[0, 1]
                    print(f"PC{PCi:d} Before correlation {Rcorr_init:.4f}")
                    phi_offset = -0.5 if Rcorr_init < -0.8 else 0.0
                    try:
                        params, params_covariance = curve_fit(test_func, xarr, curv,
                                                              p0=[curv.max(), PCi / 2, phi_offset, 0],
                                                              bounds=([0, 0, -1, curv.min()],
                                                                      [np.inf, np.inf, 1, curv.max()]),
                                                              maxfev=1E4)
                        A, omega, phi, bsl = params
                        Rcorr = np.corrcoef(test_func(xarr, *params), curv)[0, 1]
                        print(f"Fitted curve {A:.1f} cos(2pi {omega:.3f}(x + {phi:.3f})) + {bsl:.3f}")
                        print(f"PC{PCi:d} Correlation {Rcorr:.4f} ")
                    except RuntimeError:
                        A, omega, phi, bsl = np.nan, np.nan, np.nan, np.nan
                        Rcorr = 0
                    param_col.append(EasyDict(PCi=PCi, A=A, omega=omega, phi=phi, bsl=bsl,
                                              Rcorr_init=Rcorr_init, Rcorr=Rcorr))

                cosfit_df = pd.DataFrame(param_col)
                cosfit_df["expvar"] = expvar_ratio[:74]
                cosfit_df.to_csv(join(Droot, unitdir, "CholeskyCMAES_meanPCA_cosfit_%s_%s_%03d_%05d.csv" \
                                      % (netname, layername, channum, RND)))

    return data  # maxobj_df, cleanobj_df, runtime_df, codenorm_df, optimlist

dataroot = r"E:\Cluster_Backup\cma_optim_cmp"
data = load_trajectory_data_cma(dataroot, sumdir="summary")
#%%
def load_trajectory_PCAdata_cma(Droot, sumdir="summary"):
    subdirs = os.listdir(Droot)
    df_col = {}
    meta_col = {}
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
        csvpaths = glob(join(Droot, unitdir, "CholeskyCMAES_PCA_cosfit_*.csv"))
        df_col[unitdir] = []
        meta_col[unitdir] = []
        for fpath in csvpaths:
            shortfn = os.path.split(fpath)[1]
            # CholeskyCMAES_alexnet_.classifier.Linear6_003_77176.npz
            toks = re.findall("CholeskyCMAES_PCA_cosfit_(.*)_(\.[^_]*)_(\d*)_(\d\d\d\d\d).csv", shortfn)
            assert len(toks) == 1, "cannot match pattern for %s"%shortfn
            assert toks[0][0] == netname
            assert toks[0][1] == layername
            assert channum == int(toks[0][2])
            RND = int(toks[0][3])
            expmeta = EasyDict(netname=netname, layername=layername, channum=channum,
                               noise_level=noise_level, RND=RND,
                               expdir=unitdir)
            PCA_df = pd.read_csv(fpath)
            df_col[unitdir].append(PCA_df)
            meta_col[unitdir].append(expmeta)
    return df_col, meta_col


def load_mean_trajectoryPCAdata_cma(Droot, sumdir="summary"):
    subdirs = os.listdir(Droot)
    df_col = {}
    meta_col = {}
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
        for fpath in csvpaths:
            shortfn = os.path.split(fpath)[1]
            # CholeskyCMAES_alexnet_.classifier.Linear6_003_77176.npz
            toks = re.findall("CholeskyCMAES_meanPCA_cosfit_(.*)_(\.[^_]*)_(\d*)_(\d\d\d\d\d).csv", shortfn)
            assert len(toks) == 1, "cannot match pattern for %s"%shortfn
            assert toks[0][0] == netname
            assert toks[0][1] == layername
            assert channum == int(toks[0][2])
            RND = int(toks[0][3])
            expmeta = EasyDict(netname=netname, layername=layername, channum=channum,
                               noise_level=noise_level, RND=RND,
                               expdir=unitdir)
            PCA_df = pd.read_csv(fpath)
            df_col[unitdir].append(PCA_df)
            meta_col[unitdir].append(expmeta)
    return df_col, meta_col


dataroot = r"E:\Cluster_Backup\cma_optim_cmp"
# df_col, meta_col = load_trajectory_PCAdata_cma(dataroot)
meandf_col, meta_col = load_mean_trajectoryPCAdata_cma(dataroot)

#%% Figure 3D
plt.figure(figsize=[5, 5])
PCarr = np.arange(1, 75)
plt.loglog(PCarr, meandf_col['alexnet_.classifier.Linear6_000'][0].expvar,
           lw=1, alpha=0.1, c='k', label="Empirical (N=210)")
for layer in meandf_col:
    for i in range(1):
        plt.loglog(PCarr, meandf_col[layer][i].expvar,
               lw=0.2, alpha=0.1, c='k')
plt.loglog(PCarr, 6/PCarr**2/np.pi**2,
           lw=2.5, ls="-.", alpha=0.6, label="Theory: N inf limit")
NormConst = (75**2 - 1) / 3 #1 / np.sum(1 / (1 - np.cos(np.pi * PCarr / 75)))
plt.loglog(PCarr, 1 / (1 - np.cos(np.pi * PCarr / 75)) / NormConst,
           lw=4, ls=":", color="r", alpha=0.6, label="Theory: N finite")
plt.xlabel("PC number", fontsize=14)
plt.ylabel("Explained Variance Ratio", fontsize=14)
plt.title("Explained Variance of PCs of Search Trajectory\n(AlexNet 8 layers)", fontsize=18)
plt.legend()
plt.savefig(join(figdir, "PCA_expvar_theory_cmp.png"))
plt.savefig(join(figdir, "PCA_expvar_theory_cmp.pdf"))
plt.show()

#%% Figure 3B
plt.figure(figsize=[5, 5])
PCarr = np.arange(1, 75)
plt.plot(PCarr, meandf_col['alexnet_.classifier.Linear6_000'][0].omega,
           lw=1, alpha=0.1, c='k', label="Empirical (N=210)")
for layer in meandf_col:
    for i in range(1):
        plt.plot(PCarr, meandf_col[layer][i].omega,
               lw=0.5, alpha=0.3, c='k')
plt.plot(PCarr, PCarr / 2,
           lw=1.5, ls="-.", alpha=0.9, c="r", label="Theory")
plt.xlabel("PC number", fontsize=14)
plt.ylabel("Angular frequency", fontsize=14)
plt.title("Angular frequency of PC projections\n(AlexNet 8 layers)", fontsize=16)
plt.legend()
plt.savefig(join(figdir, "PCA_projcurv_freq_theory_cmp.png"))
plt.savefig(join(figdir, "PCA_projcurv_freq_theory_cmp.pdf"))
plt.show()

#%% Figure 3C Line version
plt.figure(figsize=[5, 5])
PCarr = np.arange(1, 75)
plt.plot(PCarr, meandf_col['alexnet_.classifier.Linear6_000'][0].Rcorr**2,
           lw=1, alpha=0.1, c='k', label="Empirical (N=210)")
for layer in meandf_col:
    for i in range(1):
        plt.plot(PCarr, meandf_col[layer][i].Rcorr**2,
               lw=0.3, alpha=0.1, c='k')
        plt.plot(PCarr, meandf_col[layer][i].Rcorr_init**2,
               lw=0.3, alpha=0.1, c='b')
plt.xlabel("PC number", fontsize=14)
plt.ylabel("R square", fontsize=14)
plt.title("Cosine Fitting R2 of PC projections\n(AlexNet 8 layers)", fontsize=16)
plt.legend()
plt.savefig(join(figdir, "PCA_cos_fitting_R2_init_cmp.png"))
plt.savefig(join(figdir, "PCA_cos_fitting_R2_init_cmp.pdf"))
plt.show()
#%% Figure 3C Shaded version
plt.figure(figsize=[5, 5])
PCarr = np.arange(1, 75)
R2corr_arr = []
R2init_arr = []
for layer in meandf_col:
    for i in range(5):
        R2corr_arr.append(meandf_col[layer][i].Rcorr**2)
        R2init_arr.append(meandf_col[layer][i].Rcorr_init**2)
R2corr_arr = np.array(R2corr_arr)
R2init_arr = np.array(R2init_arr)
plt.plot(PCarr, np.median(R2corr_arr, axis=0), color="k", lw=2)
plt.plot(PCarr, np.median(R2init_arr, axis=0), color="b", lw=2)
plt.fill_between(PCarr, np.percentile(R2corr_arr, 5, axis=0),
                    np.percentile(R2corr_arr, 95, axis=0),
                    label="fitting phase and freq", color="k", alpha=0.3)
plt.fill_between(PCarr, np.percentile(R2init_arr, 5, axis=0),
                    np.percentile(R2init_arr, 95, axis=0),
                    label="fixing phase and freq", color="b", alpha=0.3)
plt.hlines(0.8, 1, 75, lw=2, ls=":", color="r")
plt.xlabel("PC number", fontsize=14)
plt.ylabel("R square", fontsize=14)
plt.title("Cosine Fitting R2 of Proj Curves\n(AlexNet 8 layers, N=1050)", fontsize=16)
plt.legend()
plt.savefig(join(figdir, "PCA_cos_fitting_R2_init_cmp_shaded.png"))
plt.savefig(join(figdir, "PCA_cos_fitting_R2_init_cmp_shaded.pdf"))
plt.show()

#%% Figure 3A Example
unitlabel = "alexnet_.classifier.Linear6_000"
unitdir = join(r"E:\Cluster_Backup\cma_optim_cmp", unitlabel)
trajdata = np.load(join(unitdir, "CholeskyCMAES_%s_12731.npz"%unitlabel))
meanPCdata = np.load(join(unitdir, "CholeskyCMAES_meanPCA_coef_%s_12731.npz"%unitlabel))
PCdata = np.load(join(unitdir, "CholeskyCMAES_PCA_coef_%s_12731.npz"%unitlabel))
#%%
cmap = plt.get_cmap('hsv')
PCnum2plot = 8
plt.figure(figsize=[5, 5])
PCarr = np.arange(1, 76)
for PCi in range(PCnum2plot):
    plt.plot(PCarr, meanPCdata['PCcoefs_mean'][:, PCi], label="PC%d"%(PCi+1),
             alpha=0.7, lw=1.5, color=cmap(1.0/(PCnum2plot+2)*PCi))
plt.legend(loc="lower left")
plt.xlabel("Generations", fontsize=14)
plt.ylabel("Projected Coefficient", fontsize=14)
plt.title("AlexNet FC8 000 ", fontsize=16)
plt.savefig(join(figdir, "Example_PC_cos_curves_fc8_01.png"))
plt.savefig(join(figdir, "Example_PC_cos_curves_fc8_01.pdf"))
plt.show()
#%%
figh = plt.figure(figsize=[6.5,5])
# sns.heatmap(np.abs(meanPCdata['PCvecs']@PCdata['PCvecs'].T), )
# xticklabels=range(1,51,5), yticklabels=range(1,51,5),)
cosangmat = np.abs(meanPCdata['PCvecs'][:50,:]@PCdata['PCvecs'].T)
plt.matshow(cosangmat, cmap="rocket", fignum=0)
plt.colorbar()
plt.xticks(range(4,51,5),range(5,51,5))
plt.yticks(range(4,51,5),range(5,51,5))
plt.ylabel("Mean Traj PC axes", fontsize=14)
plt.xlabel("All Codes PC axes", fontsize=14)
plt.xlim([1, 50])
plt.axis("image")
plt.title("Comparison of PC axes for Mean Traj and All codes", fontsize=16)
plt.savefig(join(figdir, "Example_PC_axes_inprod_mat_fc8_01.png"))
plt.savefig(join(figdir, "Example_PC_axes_inprod_mat_fc8_01.pdf"))
plt.show()
#%%
PCmachine = PCA(2000)
codes_PC_reduced = PCmachine.fit_transform(X=trajdata["codes_all"])
plt.figure()
plt.loglog(np.arange(1,2001), PCmachine.explained_variance_ratio_)
plt.show()

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

plt.show()

projtraj = meanPCdata['PCcoefs_mean']
K = 6
figh, axs = plt.subplots(K, K, figsize=[8,8], )
for i in range(K+1):
    for j in range(i):
        colorline(projtraj[:,j], projtraj[:,i], ax=axs[i-1, j],
                  cmap=plt.get_cmap('jet'), linewidth=2, alpha=0.6)
        # timearr = np.arange(len(projtraj[:,j]))
        # axs[i-1, j].plot(projtraj[:,j], projtraj[:,i], c="k")
        # timearr = np.arange(len(projtraj[:,j]))
        # axs[i-1, j].scatter(projtraj[:,j], projtraj[:,i], c=timearr, marker='o',)
        if j > 0:
            axs[i - 1, j].set_yticklabels([])
        if j == 0:
            axs[i - 1, j].set_ylabel(f"PC {i+1}")
        if i < K:
            axs[i-1, j].set_xticklabels([])
        if i == K:
            axs[i - 1, j].set_xlabel(f"PC {j+1}")
        axs[i-1, j].axis("auto")

for i in range(K):
        for j in range(i+1, K):
            axs[i, j].axis("off")
figh.tight_layout()
plt.savefig(join(figdir, "Example_Lissajous_curves_fc8_01_time.png"))
plt.savefig(join(figdir, "Example_Lissajous_curves_fc8_01_time.pdf"))
plt.show()


#%%
df_col, meta_col = load_trajectory_PCAdata_cma(dataroot)
#%%
layerlist = ['alexnet_.features.ReLU4_000',
             'alexnet_.features.ReLU7_000',
             'alexnet_.features.ReLU9_000',
             'alexnet_.features.ReLU11_000',
             'alexnet_.classifier.ReLU2_000',
             'alexnet_.classifier.ReLU5_000',
             'alexnet_.classifier.Linear6_000',]

plt.figure()
for layer in layerlist:
    plt.plot(df_col[layer].expvar.cumsum(), alpha=0.7)
plt.show()
#%%
plt.figure()
for layer in df_col:
    # plt.plot(df_col[layer][1].expvar.cumsum(), alpha=0.7)
    plt.plot(df_col[layer][4].Rcorr, alpha=0.1)
plt.show()
#%% log log plot of the explained variance or the Activation ratio
plt.figure()
PCarr = np.arange(1, 51)
for layer in df_col:
    # plt.plot(df_col[layer][1].expvar.cumsum(), alpha=0.7)
    for i in range(1):
        # plt.plot(np.log10(PCarr), np.log10(df_col[layer][i].expvar), alpha=0.1)
        plt.loglog(PCarr, df_col[layer][i].expvar, linewidth=0.3, alpha=0.3)
# plt.plot(np.log10(PCarr), np.log10(6/PCarr**2/np.pi**2), alpha=0.1, linestyle="-.")
plt.loglog(PCarr, 6/PCarr**2/np.pi**2)
plt.loglog(PCarr, 1/(1/np.arange(1,4097)**2).sum()/PCarr**2)

plt.show()
#%%
plt.figure()
for layer in df_col:
    # plt.plot(df_col[layer][1].expvar.cumsum(), alpha=0.7)
    for i in range(1):
        plt.plot(np.log10(np.arange(1, 51)),
                 np.log10(df_col[layer][i].A), alpha=0.1)
plt.ylim([-1.8, 2.5])
plt.show()
#%%
layerlist = ['alexnet_.features.ReLU4_000',
             'alexnet_.features.ReLU7_000',
             'alexnet_.features.ReLU9_000',
             'alexnet_.features.ReLU11_000',
             'alexnet_.classifier.ReLU2_000',
             'alexnet_.classifier.ReLU5_000',
             'alexnet_.classifier.Linear6_000',]

plt.figure()
for layer in layerlist:
    plt.plot(df_col[layer[:-1]+"1"].expvar.cumsum(), alpha=0.7)
    print(df_col[layer[:-1]+"1"].expvar.cumsum().iloc[-1])
plt.show()

#%%
codes_all = data["codes_all"]
generations = data["generations"]
scores_all = data["scores_all"]
cleanscores_all = data["cleanscores_all"]
meancodes = np.array([codes_all[generations == i,:].mean(axis=0) for i in range(generations.max()+1)])
#%%
PCmachine = PCA(n_components=1001).fit(codes_all)
PCcoefs = PCmachine.transform(codes_all)
PCcoefs_mean = PCmachine.transform(meancodes)
#%%
plt.figure()
PCarr = np.arange(1, 1002)
plt.loglog(PCarr, PCmachine.explained_variance_ratio_)
plt.loglog(PCarr, 6/PCarr**2/np.pi**2)
plt.loglog(PCarr, 1/(1/np.arange(1,4097)**2).sum()/PCarr**2)
plt.show()
#%% If we fit the PCA on the mean codes
PCmachine2 = PCA(n_components=74).fit(meancodes)
PCcoefs_mean2 = PCmachine2.transform(meancodes)
#%% Then the explained variance could be exactly predicted by the equation!
plt.figure()
PCarr = np.arange(1, 75)
plt.loglog(PCarr, 6/PCarr**2/np.pi**2, lw=2.5, alpha=0.6, label="Theory: N inf limit")
NormConst = (75**2 - 1) / 3 #1 / np.sum(1 / (1 - np.cos(np.pi * PCarr / 75)))
plt.loglog(PCarr, 1 / (1 - np.cos(np.pi * PCarr / 75)) / NormConst,
           lw=2.5, ls=":", color="r", alpha=0.6, label="Theory: N finite")
plt.loglog(PCarr, PCmachine2.explained_variance_ratio_, lw=2.5, alpha=0.5, c="k", label="Empirical")
# plt.loglog(PCarr, 1/(1/np.arange(1, 4097)**2).sum()/PCarr**2)
plt.legend() # [, "Theory: N inf limit", "Theory: N finite"]
plt.xlabel("PC number")
plt.ylabel("Explained Variance Ratio")
plt.show()
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

#%% Noise Sinusoidal structure
datadir = r"F:\insilico_exps\noise_optim_ctrl\CholeskyCMAES"
#%% Loading up the noise driven evolution data
# "CholeskyCMAES_meanPCA_coef_rep02681.npz"
expvar_col = []
PCcoefs_col = []
npzpaths = glob(join(datadir,"CholeskyCMAES_meanPCA_coef_rep*.npz"))
for fpath in npzpaths:
    shortfn = os.path.split(fpath)[1]
    # CholeskyCMAES_alexnet_.classifier.Linear6_003_77176.npz
    toks = re.findall("CholeskyCMAES_meanPCA_coef_rep(\d\d\d\d\d).npz", shortfn)
    RND = int(toks[0][0])
    data = np.load(fpath)
    expvar_col.append(data['expvar_ratio'].copy())
    PCcoefs_col.append(data['PCcoefs_mean'].copy())
#%%
# plt.figure(figsize=[5, 5])
# plt.plot(data['PCcoefs_mean'][:, :8])
# plt.xlabel("Generations",fontsize=14)
# plt.ylabel("Projection Coefficient",fontsize=14)
# plt.title("Example PC Projections\n(Noise Driven)",fontsize=14)
# plt.savefig(join(figdir, "Example_PC_cos_curves_noise001.png"))
# plt.savefig(join(figdir, "Example_PC_cos_curves_noise001.pdf"))
# plt.legend(["PC%d" % i for i in range(1, 9)])
# plt.show()

cmap = plt.get_cmap('hsv')
PCnum2plot = 8
plt.figure(figsize=[5, 5])
PCarr = np.arange(1, 76)
for PCi in range(PCnum2plot):
    plt.plot(PCarr, data['PCcoefs_mean'][:, PCi], label="PC%d"%(PCi+1),
             alpha=0.7, lw=1.5, color=cmap(1.0/(PCnum2plot+2)*PCi))
plt.legend(loc="lower left")
plt.xlabel("Generations", fontsize=14)
plt.ylabel("Projected Coefficient", fontsize=14)
plt.title("Example PC Projections\n(Noise Driven)",fontsize=16)
plt.savefig(join(figdir, "Example_PC_cos_curves_noise001.png"))
plt.savefig(join(figdir, "Example_PC_cos_curves_noise001.pdf"))
plt.show()
#%%
"""Supplementary Figure Compare the noise driven evolution and 
Real evolution per explained varience per PC
"""
plt.figure(figsize=[5, 5])
PCarr = np.arange(1, 75)
plt.loglog(PCarr, meandf_col['alexnet_.classifier.Linear6_000'][0].expvar,
           lw=0.0, alpha=0.2, c='r', label="Evolution (N=1050)")
for layer in meandf_col:
    for i in range(5):
        plt.loglog(PCarr, meandf_col[layer][i].expvar,
               lw=0.2, alpha=0.2, c='r')
plt.loglog(PCarr, expvar_col[0][:-1],
               color='k', alpha=0.2, lw=0.2, label="Noise Evol (N=100)")
for i in range(len(expvar_col)):
    plt.loglog(PCarr, expvar_col[i][:-1],
               color='k', alpha=0.2, lw=0.2)
plt.xlabel("PC number", fontsize=14)
plt.ylabel("Explained Variance Ratio", fontsize=14)
plt.title("Explained Variance of PCs of Evol Trajectory\nAlexNet vs Noise Driven", fontsize=18)
plt.legend(fontsize=14)
plt.savefig(join(figdir, "PCA_expvar_evol_noise_cmp.png"))
plt.savefig(join(figdir, "PCA_expvar_evol_noise_cmp.pdf"))
plt.legend()
plt.show()
#%%
expvar_evol = np.array([meandf_col[layer][i].expvar for i in range(5) for layer in meandf_col])
expvar_nois = np.array([expvar_col[i][:-1] for i in range(len(expvar_col))])
#%%
from scipy.stats import ttest_ind,ttest_rel
pval_col = [ttest_ind(expvar_evol[:,i],expvar_nois[:,i]).pvalue for i in range(74)]
