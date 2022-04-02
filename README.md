# Activation Maximization Optimizer Development

## Background
This project aims at benchmarking, analyzing and improving performance for evolutionary optimizers for activation maximization. 

We conducted two rounds of *in silico* benchmark of over 15 gradient free optimizers, using multiple CNN architecture, multiple depth and multiple noise level. We found that `CMA` type optimizer performed consistently well for this problem. To understand why, we conducted a series of analysis on the mechanism of CMA optimizer. We found that its success is less related to its adaptation of covariance matrix. In contrast, it's related to the implicit angular step size decay during evolution. 

We further found some intriguing geometric properties of the evolution trajectory which are comparable to high-dimensional (guided) random walk: the trajectory exhibit sinusoidal structure when projected onto top PC space; the distance travelled in latent space is proportional to square root of step number; the trajectory preferably aligns with the top eigen dimensions of the underlying image manifold. 

Given these analyses, we further developped a new optimizer `SphereCMA` which leverages the spherical geometry of the space and performed better than the original CMA optimizer.

## Try it out!

```python
import numpy as np
from core.insilico_exps import ExperimentEvolution
from core.Optimizers import CholeskyCMAES, ZOHA_Sphere_lr_euclid, Genetic, pycma_optimizer

tmpsavedir = "" # Temporary save directory

# load optimizer
optim = CholeskyCMAES(4096, population_size=40, init_sigma=2.0, Aupdate_freq=10, init_code=np.zeros([1, 4096]))
# un-comment to use our new one! 
# optim = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20, lr=1.5, sphere_norm=300)
# optim.lr_schedule(n_gen=75, mode="exp", lim=(50, 7.33) ,)
explabel, model_unit = "alexnet_fc8_1", ("alexnet", ".classifier.Linear6", 1)
Exp = ExperimentEvolution(model_unit, savedir=tmpsavedir, explabel=explabel, optimizer=optim)
# run evolutions
Exp.run(optim.get_init_pop())
Exp.visualize_best()
Exp.visualize_trajectory()
Exp.save_last_gen()
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1F5gJjzrNHAgRuIzGmenzqiiChtwk2ZXu?usp=sharing) This tutorial walks through Evolution experiments, the basic properties of trajectories (PC structure, etc.) and our improved spherical CMA optimizer. 

## Structure of Repo

* Root directory contains the `python` scripts for large scale benchmarking experiments
	* `scripts` contains the bash scripts for running large scale experiments on cluster. 
* `core` contains the major toolkit, the key classes and utility functions for activation maximization experiments.
* `analysis` contains the scripts for reproducing the statistics and figures in the paper. 
* `summary` contains raw `csv` data for the major performance results for our *in silico* benchmarks. 

## Dependency

* `pip install cma==3.0.3`
* `pip install nevergrad==0.4.2.post5`
* `pip install pytorch_pretrained_biggan` 
<!-- `conda install --channel cma-es cma` 
`pip install pyswarms`  -->
 
**Versions**: `nevergrad 0.4.3` seems to change the default behavior, all our benchmarks were done using the following version of `cma` and `nevergrad`

* `cma.__version__=='3.0.3  $Revision: 4430 $ $Date: 2020-04-21 01:19:04 +0200 (Tue, 21 Apr 2020) $'`
* `ng.__version__=='0.4.2.post5'`

