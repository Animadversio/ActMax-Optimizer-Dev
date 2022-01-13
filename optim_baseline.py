# Playground for reference optimizers
#
import cma
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
# https://github.com/ljvmiranda921/pyswarms
#%%
#%%
import torch
import numpy as np
from core.GAN_utils import upconvGAN
from core.CNN_scorers import TorchScorer
G = upconvGAN("fc6")
G.eval().cuda()
scorer = TorchScorer("resnet50")
scorer.select_unit((None,'.Linearfc',1))

#%% Test out these optimizers for out function
def objfunc(code):
    return -scorer.score(G.visualize_batch_np(code))

bounds = (-6*np.ones(4096), 6*np.ones(4096))
# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.7}
bh_strategy = "nearest" # "random" # "nearest is much better than random
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=40, dimensions=4096, bounds=bounds, bh_strategy=bh_strategy, options=options)
# Perform optimization
best_cost, best_pos = optimizer.optimize(objfunc, iters=100)
#%%
def objfunc(code):
    return -scorer.score(G.visualize_batch_np(code[np.newaxis,:]))
es = cma.CMAEvolutionStrategy(4096 * [0], 3.0)
es.optimize(objfunc, iterations=100, maxfun=4000)
#   100   2800 -3.126652526855469e+01 1.0e+00 2.62e+00  3e+00  3e+00 2:30.7
#%%
es = cma.CMAEvolutionStrategy(4096 * [0], 3.0)
es.optimize(objfunc, maxfun=4000)
#
#%%
es = cma.CMAEvolutionStrategy(4096 * [0], 2.0)
es.optimize(objfunc, iterations=100, maxfun=4000)
# 100   2800 -3.266315460205078e+01 1.0e+00 1.74e+00  2e+00  2e+00 2:46.8
#%%
es = cma.CMAEvolutionStrategy(4096 * [0], 2.0)
es.optimize(objfunc, maxfun=4000)#iterations=100,
#   141   3948 -3.340768432617188e+01 1.0e+00 1.69e+00  2e+00  2e+00 4:12.0
#%%
es = cma.CMAEvolutionStrategy(4096 * [0], 1.5)
es.optimize(objfunc, maxfun=4000)#iterations=100,
#   141   3948 -3.605181503295898e+01 1.0e+00 1.26e+00  1e+00  1e+00 4:16.4
#%%
es = cma.CMAEvolutionStrategy(4096 * [0], 1.0)
es.optimize(objfunc, iterations=100, maxfun=4000)
#   100   2800 -3.093707466125488e+01 1.0e+00 8.72e-01  9e-01  9e-01 2:49.4
#%%
from core.insilico_exps import ExperimentEvolution
Exp = ExperimentEvolution(("resnet50", '.Linearfc', 1))
Exp.run()
#synth img scores: mean 27.614 +- std 2.102
