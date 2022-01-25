# Playground for reference optimizers
#
import cma
import pyswarms as ps
# from pyswarms.utils.functions import single_obj as fx
import nevergrad as ng
import torch
import numpy as np
from core.GAN_utils import upconvGAN
from core.CNN_scorers import TorchScorer
# https://github.com/ljvmiranda921/pyswarms
#%%
G = upconvGAN("fc6")
G.eval().cuda()
scorer = TorchScorer("resnet50") # _linf8
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
es.optimize(objfunc, maxfun=4000)
#   141   3948 -3.340768432617188e+01 1.0e+00 1.69e+00  2e+00  2e+00 4:12.0
#%%
es = cma.CMAEvolutionStrategy(4096 * [0], 1.5)
es.optimize(objfunc, maxfun=4000)
#   141   3948 -3.605181503295898e+01 1.0e+00 1.26e+00  1e+00  1e+00 4:16.4
#%%
es = cma.CMAEvolutionStrategy(4096 * [0], 1.0)
es.optimize(objfunc, iterations=100, maxfun=4000)
#   100   2800 -3.093707466125488e+01 1.0e+00 8.72e-01  9e-01  9e-01 2:49.4
#%%
from core.insilico_exps import ExperimentEvolution
Exp = ExperimentEvolution(("resnet50", '.Linearfc', 1))
Exp.run()
# synth img scores: mean 27.614 +- std 2.102
#%%
def score_batch(z, ):
    return -scorer.score(G.visualize_batch_np(z.reshape([-1,4096])))


pop_size = 28
instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)),)
optimizer = ng.optimizers.TBPSA(parametrization=instrum, budget=2800, num_workers=10,)
optimizer.minimize(score_batch, verbosity=True, batch_mode=True)
# Updating fitness with value [-4.03109121]
#%%
pop_size = 28
instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)).set_bounds(-5, 5),)
optimizer = ng.optimizers.PSO(parametrization=instrum, budget=2800, num_workers=10,)
optimizer.minimize(score_batch, verbosity=True, batch_mode=True)
# {'optimistic': MultiValue<mean: -5.7814459800720215, count: 1,
#  'pessimistic': MultiValue<mean: -5.7814459800720215, count: 1,
#  'average': MultiValue<mean: -5.7814459800720215, count: 1,
#%% Obvious better than PSO & TBPSA which failed miserably.
instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)).set_bounds(-5, 5),)
optimizer = ng.optimizers.NGO(parametrization=instrum, budget=2800, num_workers=10,)
optimizer.minimize(score_batch, verbosity=True, batch_mode=True)
# -12.787981986999512, -10.14200987134661
#%%
instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)).set_bounds(-5, 5),)
optimizer = ng.optimizers.ES(parametrization=instrum, budget=2800, num_workers=10,)
optimizer.minimize(score_batch, verbosity=True, batch_mode=True)
#%%
instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)).set_bounds(-5, 5),)
optimizer = ng.optimizers.CMA(parametrization=instrum, budget=2800, num_workers=10,)
optimizer.minimize(score_batch, verbosity=True, batch_mode=True)
#%%
instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)).set_bounds(-5, 5),)
optimizer = ng.optimizers.DiagonalCMA(parametrization=instrum, budget=2800, num_workers=10,)
optimizer.minimize(score_batch, verbosity=True, batch_mode=True)
# -11.705142974853516; -11.705142974853516; -11.705142974853516
#%%
instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)).set_bounds(-5, 5),)
optimizer = ng.optimizers.SQPCMA(parametrization=instrum, budget=2800, num_workers=10,)
optimizer.minimize(score_batch, verbosity=True, batch_mode=True)
# -2.43425274
#%%
instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)).set_bounds(-5, 5),)
optimizer = ng.optimizers.NelderMead(parametrization=instrum, budget=2800, num_workers=1,)
optimizer.minimize(score_batch, verbosity=True, batch_mode=True)
#%%
instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)).set_bounds(-5, 5),)
optimizer = ng.optimizers.OnePlusOne(parametrization=instrum, budget=2800, num_workers=1,)
optimizer.minimize(score_batch, verbosity=True, batch_mode=True)
# -12.376550674438477 -12.376550674438477 -12.376550674438477
# not bad not worse.
#%%
pop_size = 28
instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)).set_bounds(-5, 5),)
optimizer = ng.optimizers.BO(parametrization=instrum, budget=2800, num_workers=10,)
optimizer.minimize(score_batch, verbosity=True, batch_mode=True)
# it's excruciatingly slow, cannot finish.
#%%
#%%
while optimizer.num_ask < optimizer.budget:
    xbatch = []
    for i in range(pop_size):
        x = optimizer.ask()
        xbatch.append(x.args[0])

    loss = score_batch(np.array(xbatch), ) #**x.kwargs
    for x, loss_single in zip(xbatch,loss):
        optimizer.tell(xbatch, loss_single)
    print("feval %d : %.3f+-%.3f"%(optimizer.num_ask, loss.mean(), loss.std()))
#%%
# very counter intuitive..... strange. how to ask for multiple candidates and evaluate multiple
#%%
pop_size = 28
instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)),)
optimizer = ng.optimizers.CMA(parametrization=instrum, budget=2800, num_workers=4)
optimizer.minimize(score_batch, verbosity=True)
# Updating fitness with value [-37.58397293]
#%%
pop_size = 28
instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)),)
optimizer = ng.optimizers.PSO(parametrization=instrum, budget=2800, num_workers=4)
optimizer.minimize(score_batch, verbosity=True)
#%%
pop_size = 28
instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)),)
optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=2800, num_workers=4)
optimizer.minimize(score_batch, verbosity=True)
#%%
pop_size = 28
instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)),)
optimizer = ng.optimizers.BO(parametrization=instrum, budget=2800, num_workers=4)
optimizer.minimize(score_batch, verbosity=True)
#%%
# while optimizer.num_ask < optimizer.budget:
#     xbatch = []
#     for i in range(pop_size):
#         x = optimizer.ask()
#         xbatch.append(x.args[0])
#
#     loss = score_batch(np.array(xbatch), ) #**x.kwargs
#     optimizer.tell(x, loss)
#     print("feval %d : %.3f+-%.3f"%(optimizer.num_ask, loss.mean(), loss.std()))
#
# recommendation = optimizer.provide_recommendation()
# print(recommendation.value)