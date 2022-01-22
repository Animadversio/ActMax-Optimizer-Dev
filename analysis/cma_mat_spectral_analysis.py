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
def load_format_data_cma_spectra(Droot, sumdir="summary"):
    maxobj_col = []
    cleanobj_col = []
    maxeig_col = []
    mineig_col = []
    condnum_col = []
    devi2id_col = []
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
            # "summarize_alexnet_.classifier.Linear6_002_73959.pkl"
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
            cleanobj_data = {k: subd["cleanscores_all"].max() for k, subd in data.items()}
            maxeig_data = {k: subd["maxeig"] for k, subd in data.items()}
            mineig_data = {k: subd["mineig"] for k, subd in data.items()}
            condnum_data = {k: subd["condnum"] for k, subd in data.items()}
            devi2id_data = {k: subd["devi2id"] for k, subd in data.items()}
            maxobj_data.update(expmeta)
            cleanobj_data.update(expmeta)
            maxeig_data.update(expmeta)
            mineig_data.update(expmeta)
            condnum_data.update(expmeta)
            devi2id_data.update(expmeta)
            maxobj_col.append(maxobj_data)
            cleanobj_col.append(cleanobj_data)
            maxeig_col.append(maxeig_data)
            mineig_col.append(mineig_data)
            condnum_col.append(condnum_data)
            devi2id_col.append(devi2id_data)
            optimlist = [*data.keys()]

    maxobj_df = pd.DataFrame(maxobj_col)
    cleanobj_df = pd.DataFrame(cleanobj_col)
    maxeig_df = pd.DataFrame(maxeig_col)
    mineig_df = pd.DataFrame(mineig_col)
    condnum_df = pd.DataFrame(condnum_col)
    devi2id_df = pd.DataFrame(devi2id_col)
    maxobj_df.to_csv(join(sumdir, "CMA_spectra_maxobj_summary.csv"))
    cleanobj_df.to_csv(join(sumdir, "CMA_spectra_cleanobj_summary.csv"))
    maxeig_df.to_csv(join(sumdir, "CMA_spectra_maxeig_summary.csv"))
    mineig_df.to_csv(join(sumdir, "CMA_spectra_mineig_summary.csv"))
    condnum_df.to_csv(join(sumdir, "CMA_spectra_condnum_summary.csv"))
    devi2id_df.to_csv(join(sumdir, "CMA_spectra_devi2id_summary.csv"))
    return maxobj_df, cleanobj_df, maxeig_df, mineig_df, condnum_df, devi2id_df, optimlist

dataroot = r"F:\insilico_exps\cma_optim_covmat"
maxobj_df, cleanobj_df, maxeig_df, mineig_df, condnum_df, devi2id_df, optimlist = load_format_data_cma_spectra(dataroot, sumdir="summary")
#%%
condnum_df[optimlist].describe().T
#%%
devi2id_df[optimlist].describe().T
#%%
maxobj_df[optimlist].describe().T
#%%
maxobj_df[optimlist].divide(maxobj_df[optimlist].mean(axis=1), axis=0).describe().T
#%%
cleanobj_df[optimlist].divide(cleanobj_df[optimlist].mean(axis=1), axis=0).describe().T
#%%
runtime_df[optimlist].describe(include=["sem"])


#%%
import cma
import nevergrad as ng
import torch
import numpy as np
from core.GAN_utils import upconvGAN
from core.CNN_scorers import TorchScorer
from time import time
from core.Optimizers import CholeskyCMAES, ZOHA_Sphere_lr_euclid, Genetic, pycma_optimizer
#%%
G = upconvGAN("fc6")
G.eval().cuda()
G.requires_grad_(False)
scorer = TorchScorer("alexnet") # _linf8
scorer.select_unit(("alexnet",'.features.ReLU11',5,6,6))
#%%
optim = pycma_optimizer(4096, sigma0=2.0,
                        inopts={"CMA_diagonal": False, 'popsize': 40}, maximize=True)
codes = optim.get_init_pop()  # random population on the sphere.
t0 = time()
for i in range(75):
    with torch.no_grad():
        scores = scorer.score(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda"))) #.reshape([-1,4096])
    newcodes = optim.step_simple(scores, codes)
    codes = newcodes
    print("step %d Max %.1e+1  Min 1-%.1e  Median %.1e Condition %.1f"%(i, optim.es.D.max()-1, 1-optim.es.D.min(), np.median(optim.es.D),
                                                                    optim.es.D.max() / optim.es.D.min()))

condnum = optim.es.D.max() / optim.es.D.min()
devi2eye = np.linalg.norm(optim.es.C - np.eye(4096),'fro')**2 / np.linalg.norm(optim.es.C,'fro')**2
print("Final condition number %.5f Relative deviation from identity %.3e"%(condnum, devi2eye))
# Final condition number 1.00256
t1 = time()
print(f"took {t1-t0:.3f} sec, score {scores.mean():.3f}")
#%%
optim = pycma_optimizer(4096, sigma0=2.0,
                        inopts={"CMA_diagonal": True, 'popsize': 40}, maximize=True)
codes = optim.get_init_pop()  # random population on the sphere.
t0 = time()
for i in range(75):
    with torch.no_grad():
        scores = scorer.score(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda"))) #.reshape([-1,4096])
    newcodes = optim.step_simple(scores, codes)
    codes = newcodes

    diagvec = optim.es.sigma_vec.tolist()
    print("step %d Max %.1e+1  Min %.1e  Median %.1e Condition %.1f"%(i, max(diagvec), min(diagvec), np.median(diagvec),
                                                                    max(diagvec) / min(diagvec)))

condnum = optim.es.sigma_vec.condition_number
covmat = np.diag(optim.es.sigma_vec.tolist())
#%%
devi2eye = np.linalg.norm(covmat - np.eye(4096),'fro')**2 / np.linalg.norm(covmat,'fro')**2
print("Final condition number %.5f Relative deviation from identity %.3e"%(condnum, devi2eye))
# Final condition number 1.00256
t1 = time()
print(f"took {t1-t0:.3f} sec, score {scores.mean():.3f}")
#%%
reseva, resevc = np.linalg.eigh(optim.es.C - np.eye(4096))
#%%
optim = CholeskyCMAES(4096, population_size=40, init_sigma=3.0, maximize=True, init_code=np.zeros((1,4096)))
codes = optim.get_init_pop()  # random population on the sphere.
t0 = time()
for i in range(75):
    with torch.no_grad():
        scores = scorer.score(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda"))) #.reshape([-1,4096])
    newcodes = optim.step_simple(scores, codes)
    codes = newcodes
Atsr = torch.tensor(optim.A).cuda()
Ctsr = Atsr @ Atsr.T
#%%
eigvals, eigvecs = torch.linalg.eigh(Ctsr)
#%%
eigvals = eigvals.cpu()
eigvecs = eigvecs.cpu()
print("step %d Max %.1e  Min %.1e  Median %.1e Condition %.1e"%(i, eigvals.max(), eigvals.min(), np.median(eigvals),
                                                                    eigvals.max() / eigvals.min()))

condnum = eigvals.max() / eigvals.min()
#%%
devi2eye = torch.norm(Ctsr.cpu() - torch.eye(4096),'fro')**2 / torch.norm(Ctsr,'fro')**2
print("Final condition number %.5f Relative deviation from identity %.3e"%(condnum, devi2eye))
# Final condition number 1.00256
t1 = time()
print(f"took {t1-t0:.3f} sec, score {scores.mean():.3f}")