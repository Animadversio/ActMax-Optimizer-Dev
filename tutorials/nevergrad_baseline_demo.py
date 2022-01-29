"""Demo file of optimizing a unit using from nevergrad"""
import cma
import pyswarms as ps
import nevergrad as ng
import torch
import numpy as np
from core.GAN_utils import upconvGAN
from core.CNN_scorers import TorchScorer
# https://github.com/ljvmiranda921/pyswarms
#%%
G = upconvGAN("fc6")
G.eval().cuda()
scorer = TorchScorer("resnet50_linf8") #
# scorer.select_unit((None,'.layer1.Bottleneck2',1,28,28))
# scorer.select_unit((None,'.layer2.Bottleneck3',1,14,14))
# scorer.select_unit((None,'.layer3.Bottleneck5',1,7,7))
# scorer.select_unit((None,'.layer4.Bottleneck2',1,4,4))
scorer.select_unit((None,'.Linearfc',1))
def score_batch(z, ):
    return -scorer.score(G.visualize_batch_np(z.reshape([-1,4096])))

instrum = ng.p.Instrumentation(ng.p.Array(shape=(4096,)).set_bounds(-6, 6),)
#%%
optimizer = ng.optimizers.CMA(parametrization=instrum, budget=3000, num_workers=40,)
optimizer.minimize(score_batch, verbosity=True, batch_mode=True)

