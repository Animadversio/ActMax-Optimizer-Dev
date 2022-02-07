import pickle as pkl
from core.Optimizers import ZOHA_Sphere_lr_euclid
import os
import re
from os.path import join
from glob import glob
from time import time
from easydict import EasyDict
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from sklearn.decomposition import PCA
import matplotlib as mpl
from core.montage_utils import make_grid, ToPILImage
#%%
from core.GAN_utils import upconvGAN
G = upconvGAN("fc6")
G.requires_grad_(False)
G.cuda().eval()
#%%
figdir = r"E:\OneDrive - Harvard University\GECCO2022\Figures\Evol_prototypes"
unitdir = r"E:\Cluster_Backup\ng_optim_cmp\resnet50_linf8_.Linearfc_003"
# unitdir = r"E:\Cluster_Backup\ng_optim_cmp\resnet50_linf8_.Linearfc_001"
unitdir = r"E:\Cluster_Backup\ng_optim_cmp\alexnet_.classifier.Linear6_003"
unitstr = unitdir.split("\\")[-1]
fpaths = glob(unitdir+"\\*.pkl")
#%%
runi = 0
data = pkl.load(open(fpaths[runi],'rb'))
optimorder = ['CMA','DiagonalCMA','SQPCMA','RescaledCMA','ES',
             'NGOpt','DE','TwoPointsDE',
             'PSO','OnePlusOne','TBPSA',
             'RandomSearch']
figh,axs = plt.subplots(3,4,figsize=(8,7))
for i, optimnm in enumerate(optimorder):
    img = G.visualize_batch_np(data[optimnm].bestcode[np.newaxis, :])
    plt.sca(axs.flatten()[i])
    plt.imshow(img[0].permute([1,2,0]).numpy())
    plt.axis("off")
    plt.title(optimnm+" %.1f"%data[optimnm].maxobj)
plt.tight_layout()
plt.suptitle(unitstr, fontsize=16)
plt.savefig(join(figdir,"evolvimg_montage_%s-%d.png"%(unitstr,runi)))
plt.savefig(join(figdir,"evolvimg_montage_%s-%d.pdf"%(unitstr,runi)))
plt.show()
#%%
import nevergrad as ng
ng.optimizers.ES