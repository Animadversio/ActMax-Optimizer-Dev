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
from core.GAN_utils import upconvGAN
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 200
figdir = r"E:\OneDrive - Harvard University\GECCO2022\Figures\GANsphere_geom"
#%%
import torch
from lpips import LPIPS
G = upconvGAN("fc6")
G.cuda().eval().requires_grad_(False)
Dist = LPIPS(net="squeeze", lpips=True, pretrained=True, )
Dist.cuda().requires_grad_(False)
#%%
expdir = r"E:\Cluster_Backup\cma_optim_cmp\alexnet_.classifier.Linear6_001"
data = np.load(join(expdir, "CholeskyCMAES_alexnet_.classifier.Linear6_001_89157.npz"))
#%
codes_all = data["codes_all"].copy()
generations = data["generations"].copy()
meancodes = np.array([codes_all[generations == i, :].mean(axis=0) for i in range(generations.max()+1)])
codenorms = np.linalg.norm(meancodes, axis=1)
codesnorms_all = np.linalg.norm(codes_all, axis=1)
pergen_std = np.array([codes_all[generations == i, :].std(axis=0, ddof=1).mean(axis=0)
                       for i in range(generations.max()+1)])

#%% LPIPS distance in each batch
lpipsdist_batch_col = []
for geni in tqdm(range(1, 75)):
    imgs = G.visualize(torch.tensor(codes_all[generations == geni,:]).float().cuda())
    with torch.no_grad():
        dists = Dist(imgs[0], imgs[1:]).cpu()
    lpipsdist_batch_col.append(dists.squeeze().numpy())

#%%
angdist_col = []
for i in range(1, generations.max() + 1):
    code_gen = codes_all[generations == i, :]
    distmat = cosine_distances(code_gen)
    disttriu_vec = distmat[np.triu_indices(40, 1)]
    angdist_col.append(disttriu_vec)
#%%
L2dist_col = []
for i in range(1, generations.max() + 1):
    code_gen = codes_all[generations == i, :]
    distmat = euclidean_distances(code_gen)
    disttriu_vec = distmat[np.triu_indices(40, 1)]
    L2dist_col.append(disttriu_vec)

#%%
L2dist_vec = np.array([L2dist.mean() for L2dist in L2dist_col])
cosdist_vec = np.array([angdist.mean() for angdist in angdist_col])
angdist_vec = np.arccos(1 - cosdist_vec)
lpipsdist = np.array([lpipsdist.mean() for lpipsdist in lpipsdist_batch_col])

fig,ax = plt.subplots(figsize=[5,5])
ax.plot(angdist_vec, color="red", marker="o")
ax.set_xlabel("generations", fontsize=14)
ax.set_ylabel("angular distance", color="red", fontsize=14)
ax.set_ylim(bottom=0) #[0,ax.get_ylim()[1]])
ax2=ax.twinx()
ax2.plot(lpipsdist, color="blue", marker="o")
ax2.set_ylabel("image LPIPS dist",color="blue",fontsize=14)
ax2.set_ylim(bottom=0) #[0, ax2.get_ylim()[1]])
ax3 = ax.twinx()
ax3.plot(pergen_std, color="black", marker="o")
ax3.set_xlabel("generations",fontsize=14)
ax3.set_ylabel("euclid std",color="black",fontsize=14)
ax3.set_ylim(bottom=0)  # [0,ax3.get_ylim()[1]])
plt.savefig(join(figdir, "example_batch_distance_cmp.png"))
plt.savefig(join(figdir, "example_batch_distance_cmp.pdf"))
plt.show()
# fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
#             format='jpeg',
#             dpi=100,
#             bbox_inches='tight')
#%%
cosdistmat = cosine_distances(meancodes[1:,:])
L2distmat = euclidean_distances(meancodes[1:,:])
angdistmat = np.arccos(1 - cosdistmat)
sindistmat = np.sin(np.arccos(1 - cosdistmat))
#%%

plt.show()
#%%
imgs_all = G.visualize(torch.tensor(meancodes).float().cuda())
lpipsdist_col = []
for geni in tqdm(range(0, 75)):
    with torch.no_grad():
        dists = Dist(imgs_all[geni], imgs_all[:]).cpu()
    lpipsdist_col.append(dists.squeeze().numpy())
#%%
lpipsdistmat = np.array(lpipsdist_col)
figh, axs = plt.subplots(1,3,figsize=[9,2.8])
plt.sca(axs[0])
plt.matshow(L2distmat[1:,1:], fignum=0)
plt.colorbar()
axs[0].set_title("L2 distance")
plt.sca(axs[1])
plt.matshow(angdistmat, fignum=0)
plt.colorbar()
axs[1].set_title("Angle distance")
plt.sca(axs[2])
plt.matshow(lpipsdistmat[1:,1:], fignum=0)
axs[2].set_title("Image LPIPS")
plt.colorbar()
plt.savefig(join(figdir, "example_meancode_distmat.png"))
plt.savefig(join(figdir, "example_meancode_distmat.pdf"))
plt.show()
