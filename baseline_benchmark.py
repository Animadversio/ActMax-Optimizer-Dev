import cma
import nevergrad as ng
import torch
import numpy as np
import pickle as pkl
from easydict import EasyDict
from core.GAN_utils import upconvGAN
from core.CNN_scorers import TorchScorer
from time import time
import warnings
import os
from os.path import join
from argparse import ArgumentParser
warnings.simplefilter("ignore", cma.evolution_strategy.InjectionWarning)
#%%
parser = ArgumentParser()
parser.add_argument('--units', nargs='+', type=str, required=True)
parser.add_argument('--chan_rng', nargs=2, type=int, default=[0, 10])
parser.add_argument('--rep', type=int, default=5)
parser.add_argument('--noise_lvl', type=float, default=0)
parser.add_argument('--fevalN', type=int, default=4000)

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
G = upconvGAN("fc6")
G.eval().cuda()
G.requires_grad_(False)

instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)).set_bounds(-6, 6),)

#%%
# rootdir = r"D:\Github\ActMax-Optimizer-Dev\optim_log"
# optimlist = ["NGOpt", "CMA", "RandomSearch"]
rootdir = "/scratch1/fs1/crponce/ng_optim_cmp"
optimlist = ["NGOpt", "DE", "TwoPointsDE",
             "ES", "CMA", "RescaledCMA", "DiagonalCMA", "SQPCMA", #'RealSpacePSO',
             "PSO", "OnePlusOne", "TBPSA",
             "RandomSearch"]
# savedir = r"D:\Github\ActMax-Optimizer-Dev\optim_log"
# scorer = TorchScorer("resnet50")  # _linf8
# scorer.select_unit((None, '.Linearfc', 1))
# netname, layername, unitnum = "resnet50", '.Linearfc', 1
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
    scorer = TorchScorer(netname)  # _linf8
    scorer.select_unit(unit)
    if noise_level == 0.0:
        def score_batch(z, ):
            return -scorer.score(G.visualize_batch_np(z.reshape([-1, 4096])))
    else:
        def score_batch(z, ):
            clean_score = -scorer.score(G.visualize_batch_np(z.reshape([-1, 4096])))
            noise_gain = np.maximum(0, 1 + noise_level * np.random.randn(*clean_score.shape))
            return clean_score * noise_gain

    for repi in range(repitition):
        optim_log_dict = {}
        RND = np.random.randint(100000)
        log_file = open(join(savedir, "optimlog_%s_%s_%03d_%05d.txt"%(netname, layername, channel, RND)), "w+")
        for optimname in optimlist[:]:
            optimizer = ng.optimizers.registry[optimname](parametrization=instrum, budget=budget, num_workers=40,)
            # optimizer.register_callback("ask", ng.callbacks.ProgressBar())
            t0 = time()
            optimizer.minimize(score_batch, verbosity=False, batch_mode=True,)
            t1 = time()
            maxobj   = - optimizer.current_bests["average"].mean
            bestcode = optimizer.current_bests["average"].x
            codenorm = np.linalg.norm(bestcode)
            runtime = t1 - t0
            print("Using %s optimizer.\nBest score %.3f, norm of best code %.3f.\nWall run time %.3f sec"%
                  (optimname, maxobj, codenorm, t1-t0))
            log_file.write("Using %s optimizer.\nBest score %.3f, norm of best code %.3f.\nWall run time %.3f sec\n "%
                  (optimname, maxobj, codenorm, t1-t0))
            optim_log_dict[optimname] = EasyDict(maxobj=maxobj, bestcode=bestcode,
                     codenorm=codenorm, runtime=runtime)
            np.savez(join(savedir, r"%s_%s_%s_%03d_%05d.npz") % (optimname, netname, layername, channel, RND),
                     maxobj=maxobj, bestcode=bestcode,
                     codenorm=codenorm, runtime=runtime)

        log_file.close()
        pkl.dump(optim_log_dict, open(join(savedir, "summarize_%s_%s_%03d_%05d.pkl")%(netname, layername, channel, RND), "wb"))


