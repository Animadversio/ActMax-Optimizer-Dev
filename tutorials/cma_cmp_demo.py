"""Tutorial code for cma type algorithm benchmarking"""
import cma
import nevergrad as ng
import torch
import numpy as np
from core.GAN_utils import upconvGAN
from core.CNN_scorers import TorchScorer
from time import time
#%%
G = upconvGAN("fc6")
G.eval().cuda()
G.requires_grad_(False)
# scorer = TorchScorer("resnet50") # _linf8
# scorer.select_unit((None,'.layer1.Bottleneck2',1,28,28))
# scorer.select_unit((None,'.layer2.Bottleneck3',1,14,14))
# scorer.select_unit((None,'.layer3.Bottleneck5',1,7,7))
# scorer.select_unit((None,'.layer4.Bottleneck2',1,4,4))
# scorer.select_unit((None,'.Linearfc',1))
scorer = TorchScorer("alexnet") # _linf8
scorer.select_unit(("alexnet",'.features.ReLU7',5,6,6))
# scorer.select_unit((None,'.classifier.Linear6',1))
# def score_batch(z, ):
#     return -scorer.score(G.visualize_batch_np(z.reshape([-1,4096])))
#%%
def score_batch(z, ):
    return -scorer.score(G.visualize(torch.tensor(z.reshape([-1,4096]),dtype=torch.float32).cuda()))

t0 = time()
es = cma.CMAEvolutionStrategy(4096 * [0], 1.0)
es.optimize(score_batch, maxfun=3000, verb_disp=0)
t1 = time()
es.result_pretty()
print(f"took {t1-t0:.3f} sec") # took 75.403 sec
#%% Our CMAES Cholesky Implementation
# from core.insilico_exps import ExperimentEvolution
from core.Optimizers import CholeskyCMAES, ZOHA_Sphere_lr_euclid, Genetic
t0 = time()
optim = CholeskyCMAES(4096, population_size=40, init_sigma=2.0, Aupdate_freq=10, init_code=np.zeros([1, 4096]))
codes = optim.get_init_pop()
for i in range(75):
    scores = scorer.score(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda"))) #.reshape([-1,4096])
    newcodes = optim.step_simple(scores, codes)
    codes = newcodes

t1 = time()
print(f"took {t1-t0:.3f} sec, score {scores.mean():.3f}") # Cholesky CMAES took 12.126 sec 51.5
# A update freq = 5, took 15.631 sec, score 45.065
# took 12.235 sec, score 46.170
# pop 40, took 35.686 sec, score 60.644
# took 43.626 sec, score 59.686
#%% Our ZOHA Sphere lr Optimizer
from core.Optimizers import CholeskyCMAES, ZOHA_Sphere_lr_euclid, Genetic, pycma_optimizer
t0 = time()
optim = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20, lr=1.5, sphere_norm=300)
optim.lr_schedule(n_gen=75, mode="exp", lim=(50, 7.33) ,)
codes = optim.get_init_pop() # random population on the sphere.
for i in range(75):
    with torch.no_grad():
        scores = scorer.score(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda"))) #.reshape([-1,4096])
    newcodes = optim.step_simple(scores, codes)
    codes = newcodes

t1 = time()
print(f"took {t1-t0:.3f} sec, score {scores.mean():.3f}") # Cholesky CMAES took 12.126 sec 51.5
# Sphere took 40.838 sec, score 85.419
# Sphere took 40.890 sec, score 67.830
# took 54.307 sec, score 151.371
#%%
from core.Optimizers import CholeskyCMAES, ZOHA_Sphere_lr_euclid, Genetic, pycma_optimizer
optim = pycma_optimizer(4096, sigma0=2.0, inopts={"CMA_diagonal":True, 'popsize': 40}, maximize=True)
codes = optim.get_init_pop() # random population on the sphere.
t0 = time()
for i in range(108):
    with torch.no_grad():
        scores = scorer.score(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda"))) #.reshape([-1,4096])
    newcodes = optim.step_simple(scores, codes)
    codes = newcodes

t1 = time()
print(f"took {t1-t0:.3f} sec, score {scores.mean():.3f}")
# took 59.464 sec, score 157.487
#%% Genetic algorithm
from core.Optimizers import Genetic
population_size = 28
mutation_rate = 0.25
mutation_size = 0.75
kT_multiplier = 2
n_conserve = 10
parental_skew = 0.75
optim = Genetic(4096, population_size, mutation_rate, mutation_size, kT_multiplier,
         parental_skew=parental_skew, n_conserve=n_conserve)
codes = optim._init_population # np.zeros((1,4096))
t0 = time()
for i in range(108):
    scores = scorer.score(G.visualize(torch.tensor(codes.reshape([-1,4096]),dtype=torch.float32).cuda()))
    newcodes = optim.step_simple(scores, codes)
    codes = newcodes

t1 = time()
print(f"took {t1-t0:.3f} sec, score {scores.mean():.3f}") # Cholesky CMAES took 12.126 sec 51.5
# took 10.494 sec, score 15.566
# took 10.369 sec, score 25.768
# took 7.550 sec, score 22.590
#%%
t0 = time()
es = cma.CMAEvolutionStrategy(4096 * [0], 2.0, inopts={"CMA_diagonal":True})
for i in range(108):
    codes = es.ask()
    scores = scorer.score(G.visualize(torch.tensor(codes,dtype=torch.float32).cuda()))
    es.tell(codes, -scores, )

t1 = time()
print(f"took {t1-t0:.3f} sec, score {scores.mean():.3f}") # Diagonal CMAES took 10.446 sec, -2.663668e+01 -2.688371e+01
#%% Why CMA stops working????
t0 = time()
es = cma.CMAEvolutionStrategy(4096 * [0], 1.0, inopts={"CMA_diagonal":True})
for i in range(108):
    codes = es.ask()
    scores = scorer.score(G.visualize(torch.tensor(codes, dtype=torch.float32).cuda()))
    es.tell(codes, -scores, )
    print(f"score {scores}")


t1 = time()
print(f"took {t1-t0:.3f} sec, score {scores.mean():.3f}") # Original CMAES took 50.975 sec -2.025042e+01 -2.090930e+01
#%%
import nevergrad as ng
def score_batch(z, ):
    return -scorer.score(G.visualize(torch.tensor(z, dtype=torch.float32, device="cuda")))

instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)).set_bounds(-6, 6),)
optimizer = ng.optimizers.CMA(parametrization=instrum, budget=3000, num_workers=40,)
optimizer.minimize(score_batch, verbosity=True, batch_mode=True)
#%%
#resnet50  took 79.861 sec
#alexnet  took 75.403 sec
# #%%
# t0 = time()
# nfeval = 0
# ngeneration = 0
# es = cma.CMAEvolutionStrategy(4096 * [0], 1.0, inopts={"CMA_diagonal":True})
# while nfeval < 3000:
#     codes = es.ask()
#     scores = score_batch(np.array(codes))
#     es.tell(codes, scores)
#     nfeval += len(codes)
#     ngeneration += 1
# t1 = time()
# es.result_pretty()
# print(f"took {t1-t0:.3f} sec") #Resnet50  took 79.861 sec
# # alexnet 93.853 sec  -5.283892e+01
# # alexnet diagonal  54.514 sec  -6.027345e+01

# #%%
# t0 = time()
# es = cma.CMAEvolutionStrategy(4096 * [0], 1.0, inopts={"CMA_diagonal":True})
# es.optimize(score_batch, maxfun=3000)
# t1 = time()
# es.result_pretty()
# print(f"took {t1-t0:.3f} sec")
# alexnet  took 66.766 sec