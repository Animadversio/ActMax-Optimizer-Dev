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
pd.options.display.max_columns = 10
pd.options.display.max_colwidth=200
#%%
def load_format_data_cma(Droot, sumdir="summary"):
    maxobj_col = []
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
    maxobj_df.to_csv(join(sumdir, "CMA_benchmark_maxobj_summary.csv"))
    runtime_df.to_csv(join(sumdir, "CMA_benchmark_runtime_summary.csv"))
    codenorm_df.to_csv(join(sumdir, "CMA_benchmark_codenorm_summary.csv"))
    return maxobj_df, runtime_df, codenorm_df, optimlist

dataroot = r"E:\Cluster_Backup\cma_optim_cmp"
maxobj_df, runtime_df, codenorm_df, optimlist = load_format_data_cma(dataroot, sumdir="summary")
#%%
maxobj_df[optimlist].describe()
#%%
runtime_df[optimlist].describe()
