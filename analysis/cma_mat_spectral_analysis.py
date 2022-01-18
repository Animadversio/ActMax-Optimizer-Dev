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
for i in range(108):
    with torch.no_grad():
        scores = scorer.score(G.visualize(torch.tensor(codes, dtype=torch.float32, device="cuda"))) #.reshape([-1,4096])
    newcodes = optim.step_simple(scores, codes)
    codes = newcodes
    print("step %d Max %.1e  Min %.1e  Median %.1e Condition %.1f"%(i, optim.es.D.max(), optim.es.D.min(), np.median(optim.es.D),
                                                                    optim.es.D.max() / optim.es.D.min()))
#%% 
condnum = optim.es.D.max() / optim.es.D.min()
devi2eye = np.linalg.norm(optim.es.C - np.eye(4096),'fro')**2 / np.linalg.norm(optim.es.C,'fro')**2
print("Final condition number %.5f Relative deviation from identity %.3e"%(condnum, devi2eye))
# Final condition number 1.00256
t1 = time()
print(f"took {t1-t0:.3f} sec, score {scores.mean():.3f}")
#%%
reseva, resevc = np.linalg.eigh(optim.es.C - np.eye(4096))