"""
Analysis script for comparing performance of different CMA algorithms.
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
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 200
#%%
def load_format_data_cma(Droot, sumdir="summary"):
    maxobj_col = []
    cleanobj_col = []
    runtime_col = []
    codenorm_col = []
    subdirs = os.listdir(Droot)
    for unitdir in tqdm(subdirs):
        if "alexnet" not in unitdir:
            continue
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
            if "CholeskyCMAES" not in data:
                continue
            maxobj_data = {k: subd["maxobj"] for k, subd in data.items()}
            cleanobj_data = {k: subd["cleanscores_all"].max() for k, subd in data.items()}
            runtime_data = {k: subd["runtime"] for k, subd in data.items()}
            codenorm_data = {k: subd["codenorm"] for k, subd in data.items()}
            maxobj_data.update(expmeta)
            cleanobj_data.update(expmeta)
            runtime_data.update(expmeta)
            codenorm_data.update(expmeta)
            maxobj_col.append(maxobj_data)
            cleanobj_col.append(cleanobj_data)
            runtime_col.append(runtime_data)
            codenorm_col.append(codenorm_data)
            optimlist = [*data.keys()]

    maxobj_df = pd.DataFrame(maxobj_col)
    cleanobj_df = pd.DataFrame(cleanobj_col)
    runtime_df = pd.DataFrame(runtime_col)
    codenorm_df = pd.DataFrame(codenorm_col)
    maxobj_df.to_csv(join(sumdir, "CMA_benchmark_maxobj_summary.csv"))
    cleanobj_df.to_csv(join(sumdir, "CMA_benchmark_cleanobj_summary.csv"))
    runtime_df.to_csv(join(sumdir, "CMA_benchmark_runtime_summary.csv"))
    codenorm_df.to_csv(join(sumdir, "CMA_benchmark_codenorm_summary.csv"))
    return maxobj_df, cleanobj_df, runtime_df, codenorm_df, optimlist

dataroot = r"E:\Cluster_Backup\cma_optim_cmp"
maxobj_df, cleanobj_df, runtime_df, codenorm_df, optimlist = load_format_data_cma(dataroot, sumdir="summary")

#%% Create the normalized clean score dataframe
normobj_df = cleanobj_df.copy()
layers = sorted(cleanobj_df.layershortname.unique())
for layer in layers:
    for channum in cleanobj_df.channum.unique():
        msk = (cleanobj_df.layershortname == layer)\
            & (cleanobj_df.channum == channum)
        subtb = cleanobj_df[msk]
        normalizer = subtb[optimlist].max().max()
        # normalize to the highest clean score ever achieved for this unit
        normobj_df.loc[msk, optimlist] = normobj_df.loc[msk, optimlist] / normalizer
#%%
normobj_df.to_csv(join("summary", "CMA_benchmark_normcleanobj_summary.csv"))
#%%
maxobj_df[optimlist].describe()
#%%
cleanobj_df[optimlist][cleanobj_df.noise_level==0.0].describe().T
cleanobj_df[optimlist][cleanobj_df.noise_level==0.2].describe().T
cleanobj_df[optimlist][cleanobj_df.noise_level==0.5].describe().T
#%% Summarize the layer-wise activations
layerrenamedict = { '.features.ReLU4':"conv2",
                    '.features.ReLU7':"conv3",
                    '.features.ReLU9':"conv4",
                    '.features.ReLU11':"conv5",
                    '.classifier.ReLU5':"fc6",
                    '.classifier.ReLU2':"fc7",
                    '.classifier.Linear6':"fc8",}

layerrename_f = lambda s: layerrenamedict[s]
cleanobj_df["layershortname"] = cleanobj_df.layername.apply(layerrename_f)
maxobj_df["layershortname"] = maxobj_df.layername.apply(layerrename_f)
runtime_df["layershortname"] = maxobj_df.layername.apply(layerrename_f)
sumtab_mean = cleanobj_df.groupby(["layershortname", "noise_level"]).mean()[optimlist]
sumtab_sem = cleanobj_df.groupby(["layershortname", "noise_level"]).sem()[optimlist]
noisesumtab_mean = maxobj_df.groupby(["layershortname", "noise_level"]).mean()[optimlist]
noisesumtab_sem = maxobj_df.groupby(["layershortname", "noise_level"]).sem()[optimlist]
sumtab_mean.to_csv(join("summary", "CMA_benchmark_export_summary.csv"))
sumtab_sem.to_csv(join("summary", "CMA_benchmark_export_summary_sem.csv"))
noisesumtab_mean.to_csv(join("summary", "CMA_benchmark_export_maxobj_summary.csv"))
noisesumtab_sem.to_csv(join("summary", "CMA_benchmark_export_maxobj_summary_sem.csv"))
#%% Runtime synopsis table
runtime_synoptab = pd.concat((runtime_df[optimlist].mean(),runtime_df[optimlist].sem()),axis=1).T
runtime_synoptab.to_csv(join("summary", "CMA_benchmark_export_runtime.csv"))
#%% Statistics comparison
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp

cleanobj_df.groupby(["layershortname", "noise_level"]).mean()[optimlist]
#%%
maxobj_df[optimlist].divide(maxobj_df[optimlist].mean(axis=1), axis=0).describe()
#%% statistical testx
layers = sorted(cleanobj_df.layershortname.unique())
for ns in cleanobj_df.noise_level.unique():
    for layer in layers:
        subtb = cleanobj_df[(cleanobj_df.layershortname==layer)\
                            & (cleanobj_df.noise_level==ns)]
        tval, pval = ttest_ind(subtb.CholeskyCMAES, subtb.pycmaDiagonal)
        print(f"{layer} noise {ns:.1f}, Chol vs cmaDiag {tval:.3f}({pval:.2e})")
#%%
layers = sorted(cleanobj_df.layershortname.unique())
for layer in layers:
    for ns in cleanobj_df.noise_level.unique():
        subtb = cleanobj_df[(cleanobj_df.layershortname==layer)\
                            & (cleanobj_df.noise_level==ns)]
        meanscores = subtb.mean()[optimlist]
        maxidx = meanscores.argmax()
        bestopt = optimlist[maxidx]
        bestopt_equiv = []
        for i, optnm in enumerate(optimlist):
            if i == maxidx:
                continue

            tval, pval = ttest_ind(subtb[bestopt], subtb[optnm])
            if pval > 0.001:
                bestopt_equiv.append(optnm)
            # print(f"{layer} noise {ns:.1f}, Chol vs cmaDiag {tval:.3f}({pval:.2e})")
        print(f"{layer} noise {ns:.1f}, best {bestopt}, equiv {bestopt_equiv}")


#%%
layers = sorted(cleanobj_df.layershortname.unique())
for layer in layers:
    for ns in normobj_df.noise_level.unique():
        subtb = normobj_df[(normobj_df.layershortname==layer)\
                            & (normobj_df.noise_level==ns)]
        meanscores = subtb.mean()[optimlist]
        maxidx = meanscores.argmax()
        bestopt = optimlist[maxidx]
        bestopt_equiv = []
        for i, optnm in enumerate(optimlist):
            if i == maxidx:
                continue

            tval, pval = ttest_ind(subtb[bestopt], subtb[optnm])
            if pval > 0.001:
                bestopt_equiv.append(optnm)
            # print(f"{layer} noise {ns:.1f}, Chol vs cmaDiag {tval:.3f}({pval:.2e})")
        print(f"{layer} noise {ns:.1f}, best {bestopt}, equiv {bestopt_equiv}")
#%%
normtab_mean = normobj_df.groupby(["layershortname", "noise_level"]).mean()[optimlist]
normtab_sem = normobj_df.groupby(["layershortname", "noise_level"]).sem()[optimlist]
normtab_mean.to_csv(join("summary", "CMA_benchmark_export_summary_norm.csv"))
normtab_sem.to_csv(join("summary", "CMA_benchmark_export_summary_norm_sem.csv"))
#%%
for ns in [0.0,0.2,0.5]:
    subtb = normobj_df[normobj_df.noise_level == ns]
    for i, optnm in enumerate(optimlist):
        if optnm in ["ZOHA_Sphere_exp", "Genetic","ZOHA_Sphere_inv"]:
            continue
        tval, pval = ttest_ind(subtb["ZOHA_Sphere_exp"], subtb[optnm])
        if pval < 0.005:
            print(f"All layer noise {ns:.1f}, Sph_exp vs {optnm} {tval:.3f}({pval:.2e})")

#%% Noise free and noise
figdir = r"E:\OneDrive - Harvard University\GECCO2022\Figures\CMABenchmark"
normdf_long = normobj_df.melt(id_vars=['netname', 'layername', 'channum',
                         'noise_level', 'RND', 'expdir', 'layershortname'],
                  value_vars=optimlist, var_name="optimnm", value_name="score")
layerorder = sorted(normobj_df.layershortname.unique())
for ns in [0.0, 0.2, 0.5]:
    figh = plt.figure(figsize=(6, 5))
    sns.boxplot(data=normdf_long[normdf_long.noise_level == ns], y="score", x="layershortname", hue="optimnm",
                color="red", saturation=0.4, order=layerorder)
    plt.title("CMA-style algorithm comparison AlexNet noise %.1f"%ns)
    figh.savefig(join(figdir, "cma_cmp_layerwise_ns%.1f.png"%ns))
    figh.savefig(join(figdir, "cma_cmp_layerwise_ns%.1f.pdf"%ns))
    plt.show()
#%%
normdf_long_part = normobj_df.melt(id_vars=['netname', 'layername', 'channum',
                         'noise_level', 'RND', 'expdir', 'layershortname'],
                  value_vars=['CholeskyCMAES', 'pycma', 'pycmaDiagonal',], var_name="optimnm", value_name="score")
layerorder = sorted(normobj_df.layershortname.unique())
figh = plt.figure(figsize=(4, 4))
sns.boxplot(data=normdf_long_part,
            y="score", x="layershortname", hue="optimnm",
            color="red", saturation=0.4, order=layerorder)
plt.title("AlexNet all noise")
figh.savefig(join(figdir, "cma_cmp_layerwise_part.png"))
figh.savefig(join(figdir, "cma_cmp_layerwise_part.pdf"))
plt.show()
for ns in [0.0, 0.2, 0.5]:
    figh = plt.figure(figsize=(4, 4))
    sns.boxplot(data=normdf_long_part[normdf_long_part.noise_level == ns],
                y="score", x="layershortname", hue="optimnm",
                color="red", saturation=0.4, order=layerorder)
    plt.title("AlexNet noise %.1f"%ns)
    figh.savefig(join(figdir, "cma_cmp_layerwise_part_ns%.1f.png"%ns))
    figh.savefig(join(figdir, "cma_cmp_layerwise_part_ns%.1f.pdf"%ns))
    plt.show()
#%% Export String for latex
for optnm in ["CholeskyCMAES", "pycma", "pycmaDiagonal"]:
    tval,pval = ttest_ind(normobj_df["ZOHA_Sphere_exp"], normobj_df[optnm])
    print(f"t_{{2098}}={tval:.2f},p={pval:.1e}\\times 10^{{}} for {optnm}")