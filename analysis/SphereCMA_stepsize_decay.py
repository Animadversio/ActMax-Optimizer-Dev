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
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = True
mpl.rcParams['axes.spines.top'] = False
#%%
optimizer = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20,
                              lr=1.5, sphere_norm=300)
optimizer.lr_schedule(n_gen=75, mode="exp", lim=(50, 7.33), )
rad_curv_exp = optimizer.mulist*np.sqrt(4095)
deg_curv_exp = np.rad2deg(rad_curv_exp)
genarr = np.arange(1, 76)
#%%
optimizer = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20,
                              lr=1.5, sphere_norm=300)
optimizer.lr_schedule(n_gen=75, mode="inv", lim=(50, 7.33), )
rad_curv_inv = optimizer.mulist*np.sqrt(4095)
deg_curv_inv = np.rad2deg(rad_curv_inv)

#%%
figdir = r"E:\OneDrive - Harvard University\GECCO2022\Figures\SphereCMA"
figh, ax1 = plt.subplots(figsize=[6,5])
plt.plot(genarr,deg_curv_exp,label="SphereCMA-exp")
plt.plot(genarr,deg_curv_inv,label="SphereCMA-inv")
plt.ylabel("Angular Distance (deg)",fontsize=14)
plt.xlabel("Generations",fontsize=14)
plt.legend()
ax2 = plt.twinx()
plt.plot(genarr,rad_curv_exp)
plt.plot(genarr,rad_curv_inv)
plt.ylabel("Angular Distance (rad)",fontsize=14)
plt.title("Angular Step Size Decay Function of SphereCMA",fontsize=16)
plt.savefig(join(figdir,"stepsize_decay_demo.png"))
plt.savefig(join(figdir,"stepsize_decay_demo.pdf"))
plt.show()
