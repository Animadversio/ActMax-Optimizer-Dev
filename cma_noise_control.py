"""Control Experiments of Evolution driven by pure noise."""

import nevergrad as ng
import sys
import os
from os.path import join
import torch
import numpy as np
import pickle as pkl
from easydict import EasyDict
from core.GAN_utils import upconvGAN
from core.CNN_scorers import TorchScorer
from time import time
import warnings
import cma
from sklearn.decomposition import PCA
from core.Optimizers import CholeskyCMAES, Genetic
from argparse import ArgumentParser
warnings.simplefilter("ignore", cma.evolution_strategy.InjectionWarning)
#%%
parser = ArgumentParser()
parser.add_argument('--rep', type=int, default=5)
parser.add_argument('--fevalN', type=int, default=3000)
# parser.add_argument('--RFfit', action='store_true')  # will be false if not specified.
# parser.add_argument('--imgsize', nargs=2, type=int, default=[227, 227])
# parser.add_argument('--corner', nargs=2, type=int, default=[0, 0])
args = parser.parse_args()

budget = args.fevalN
repitition = args.rep
#%%
from core.Optimizers import Genetic, CholeskyCMAES, ZOHA_Sphere_lr_euclid, pycma_optimizer

popsize = 40
def get_optimizer(optimname, opts={}):
    if optimname == "Genetic":
        population_size = 40
        mutation_rate = 0.25
        mutation_size = 0.75
        kT_multiplier = 2
        n_conserve = 10
        parental_skew = 0.75
        optimizer = Genetic(4096, population_size, mutation_rate, mutation_size, kT_multiplier,
                            parental_skew=parental_skew, n_conserve=n_conserve)
    elif optimname == "CholeskyCMAES":
        optimizer = CholeskyCMAES(4096, population_size=40, init_sigma=3.0,
                                  Aupdate_freq=10, init_code=np.zeros([1, 4096]))
    elif optimname == "pycma":
        optimizer = pycma_optimizer(4096, population_size=40, sigma0=2.0,
                                    inopts={}, maximize=True)
    elif optimname == "pycmaDiagonal":
        optimizer = pycma_optimizer(4096, population_size=40, sigma0=2.0,
                                    inopts={"CMA_diagonal": True}, maximize=True)
    elif optimname == "ZOHA_Sphere_exp":
        optimizer = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20,
                                      lr=1.5, sphere_norm=300)
        optimizer.lr_schedule(n_gen=75, mode="exp", lim=(50, 7.33), )
    elif optimname == "ZOHA_Sphere_inv":
        optimizer = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20,
                                      lr=1.5, sphere_norm=300)
        optimizer.lr_schedule(n_gen=75, mode="inv", lim=(50, 7.33), )
    else:
        raise NotImplementedError
    return optimizer

#%%
# rootdir = r"D:\Github\ActMax-Optimizer-Dev\optim_log"
if sys.platform == "linux":
    rootdir = "/scratch1/fs1/crponce/noise_optim_ctrl"
else:
    rootdir = r"D:\Github\ActMax-Optimizer-Dev\optim_log2"
os.makedirs(rootdir, exist_ok=True)
optimlist = ["CholeskyCMAES", "pycma", "pycmaDiagonal", "ZOHA_Sphere_exp"]#
for repi in range(repitition):
    RND = np.random.randint(100000)
    for optimname in optimlist:
        savedir = join(rootdir, optimname)
        os.makedirs(savedir, exist_ok=True)

        optim = get_optimizer(optimname)
        codes = optim.get_init_pop()  # random population on the sphere.
        generations = []
        scores_col = []
        cleanscore_col = []
        codes_col = []
        nGeneration = int(budget / popsize) #
        t0 = time()
        for i in range(nGeneration):
            # with torch.no_grad():
            #     cleanscores = scorer.score(G.visualize(torch.tensor(
            #         codes, dtype=torch.float32, device="cuda"))) #.reshape([-1,4096])
            # scores = add_noise(cleanscores, noise_level)
            nCodes = np.array(codes).reshape([-1, 4096]).shape[0]
            scores = np.random.randn(nCodes)
            newcodes = optim.step_simple(scores, codes, verbosity=0)
            scores_col.append(scores)
            codes_col.append(codes)
            generations.extend([i] * len(scores))
            codes = newcodes

        t1 = time()
        scores_all = np.concatenate(scores_col, axis=0)
        codes_all = np.concatenate(codes_col, axis=0)
        generations = np.array(generations)
        codes = np.array(codes)
        final_norm = np.linalg.norm(codes, axis=1).mean()
        runtime = t1 - t0
        # if optimname in ["CholeskyCMAES"]:
        np.savez(join(savedir, r"noisectrl_%s_rep%05d.npz") % (optimname, RND),
                 generations=generations, codes_all=codes_all,
                 scores_all=scores_all, runtime=runtime)

        meancodes = np.array([codes_all[generations == i, :].mean(axis=0)
                              for i in range(generations.max() + 1)])
        PCmachine_m = PCA(n_components=75).fit(meancodes)
        PCcoefs_mean = PCmachine_m.transform(meancodes)
        expvar_ratio = PCmachine_m.explained_variance_ratio_
        cumexpvar = expvar_ratio.cumsum()
        np.savez(join(savedir, "%s_meanPCA_coef_rep%05d.npz" \
                      % (optimname, RND)),
                 PCcoefs_mean=PCcoefs_mean, PCvecs=PCmachine_m.components_,
                 expvar_ratio=expvar_ratio, cumexpvar=cumexpvar
                 )

        PCmachine = PCA(n_components=75).fit(codes_all)
        PCcoefs = PCmachine.transform(codes_all)
        PCcoefs_mean = PCmachine.transform(meancodes)
        expvar_ratio = PCmachine.explained_variance_ratio_
        cumexpvar = expvar_ratio.cumsum()
        np.savez(join(savedir, "%s_PCA_coef_rep%05d.npz" \
                      % (optimname, RND)),
                PCcoefs=PCcoefs, PCcoefs_mean=PCcoefs_mean,
                PCvecs=PCmachine.components_,
                expvar_ratio=expvar_ratio, cumexpvar=cumexpvar
                )
