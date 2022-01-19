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