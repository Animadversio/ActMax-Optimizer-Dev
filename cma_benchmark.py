"""Benchmark code for comparing the cma es algorithm and the ones implemnted by us and Genetic Algorithm"""

import nevergrad as ng
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
from core.Optimizers import CholeskyCMAES, Genetic
from argparse import ArgumentParser
warnings.simplefilter("ignore", cma.evolution_strategy.InjectionWarning)
#%%
parser = ArgumentParser()
parser.add_argument('--units', nargs='+', type=str, required=True)
parser.add_argument('--chan_rng', nargs=2, type=int, default=[0, 10])
parser.add_argument('--rep', type=int, default=5)
parser.add_argument('--noise_lvl', type=float, default=0)
parser.add_argument('--fevalN', type=int, default=3000)

# parser.add_argument('--RFfit', action='store_true')  # will be false if not specified.
# parser.add_argument('--imgsize', nargs=2, type=int, default=[227, 227])
# parser.add_argument('--corner', nargs=2, type=int, default=[0, 0])
args = parser.parse_args()

budget = args.fevalN
noise_level = args.noise_lvl
netname = args.units[0]
layername = args.units[1]
chan_rng = args.chan_rng
repitition = args.rep

if len(args.units) == 5:
    centpos = (int(args.units[3]), int(args.units[4]))
    units = (netname, layername, int(args.units[2]), int(args.units[3]), int(args.units[4]))
elif len(args.units) == 3:
    centpos = None
    units = (netname, layername, int(args.units[2]))
else:
    raise ValueError("args.units should be a 3 element or 5 element tuple!")

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


def add_noise(clean_score, noise_level):
    if noise_level == 0.0:
        return clean_score
    else:
        noise_gain = np.maximum(0, 1 + noise_level * np.random.randn(*clean_score.shape))
        return clean_score * noise_gain
#%%
# rootdir = r"D:\Github\ActMax-Optimizer-Dev\optim_log"
rootdir = "/scratch1/fs1/crponce/cma_optim_cmp"
optimlist = ["Genetic", "CholeskyCMAES", "pycma", "pycmaDiagonal", "ZOHA_Sphere_exp", "ZOHA_Sphere_inv"]#
G = upconvGAN("fc6")
G.eval().cuda()
G.requires_grad_(False)

for channel in range(chan_rng[0], chan_rng[1]):
    if len(units) == 5:
        unit = (netname, layername, channel, *centpos)
        unit_lab = "%s_%s_%03d" % (netname, unit[1], unit[2])
    elif len(units) == 3:
        unit = (netname, layername, channel,)
        unit_lab = "%s_%s_%03d" % (netname, unit[1], unit[2])
    else:
        raise ValueError
    if noise_level > 0:
        unit_lab = unit_lab + "_ns%.1f" % noise_level

    savedir = join(rootdir, unit_lab)
    os.makedirs(savedir, exist_ok=True)
    #%%
    # unit = ("alexnet", ".features.ReLU11", 5, 6, 6)
    # netname = unit[0]
    scorer = TorchScorer(netname)  # _linf8
    scorer.select_unit(unit)
    for repi in range(repitition):
        RND = np.random.randint(100000)
        log_file = open(join(savedir, "optimlog_%s_%s_%03d_%05d.txt" % (netname, layername, channel, RND)), "w+")
        optim_log_dict = {}
        for optimname in optimlist:
            optim = get_optimizer(optimname)
            codes = optim.get_init_pop()  # random population on the sphere.
            generations = []
            scores_col = []
            cleanscore_col = []
            codes_col = []
            nGeneration = int(budget / popsize) #
            t0 = time()
            for i in range(nGeneration):
                with torch.no_grad():
                    cleanscores = scorer.score(G.visualize(torch.tensor(
                        codes, dtype=torch.float32, device="cuda"))) #.reshape([-1,4096])
                scores = add_noise(cleanscores, noise_level)
                newcodes = optim.step_simple(scores, codes, verbosity=0)
                cleanscore_col.append(cleanscores)
                scores_col.append(scores)
                codes_col.append(codes)
                generations.extend([i] * len(scores))
                codes = newcodes

            t1 = time()
            scores_all = np.concatenate(scores_col, axis=0)
            cleanscores_all = np.concatenate(cleanscore_col, axis=0)
            codes_all = np.concatenate(codes_col, axis=0)
            generations = np.array(generations)
            codes = np.array(codes)
            final_norm = np.linalg.norm(codes, axis=1).mean()
            bestcode = codes.mean(axis=0)
            runtime = t1 - t0
            summarystr = f"{optimname} took {t1-t0:.3f} sec, code norm {final_norm:.2f} \n score max {scores_all.max():.3f}, final mean {scores.mean():.3f},\n"+\
                         f"clean score max {cleanscores_all.max():.3f}, clean score final mean {cleanscores.mean():.3f}.\n\n"
            print(summarystr, end="")
            log_file.write(summarystr)
            optim_log_dict[optimname] = EasyDict(maxobj=scores_all.max(), bestcode=bestcode,
                                   codenorm=final_norm, runtime=runtime, scores_all=scores_all, cleanscores_all=cleanscores_all)

            if optimname == "CholeskyCMAES":
                np.savez(join(savedir, r"%s_%s_%s_%03d_%05d.npz") % (optimname, netname, layername, channel, RND),
                         generations=generations, codes_all=codes_all,
                         scores_all=scores_all, cleanscores_all=cleanscores_all, runtime=runtime)

        log_file.close()
        pkl.dump(optim_log_dict,
                 open(join(savedir, "summarize_%s_%s_%03d_%05d.pkl") % (netname, layername, channel, RND), "wb"))

