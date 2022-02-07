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

ctrldir = r"F:\insilico_exps\noise_optim_ctrl\CholeskyCMAES"
data = np.load(join(ctrldir, "noisectrl_CholeskyCMAES_rep04071.npz"))
codes_all = data["codes_all"].copy()
generations = data["generations"].copy()
meancodes = np.array([codes_all[generations == i, :].mean(axis=0)
                      for i in range(generations.max()+1)])
codenorms = np.linalg.norm(meancodes, axis=1)
codesnorms_all = np.linalg.norm(codes_all, axis=1)
pergen_std = np.array([codes_all[generations == i, :].std(axis=0, ddof=1).mean(axis=0)
                       for i in range(generations.max()+1)])
#%%
angdist_col = []
for i in range(1,generations.max()+1):
    code_gen = codes_all[generations == i, :]
    distmat = cosine_distances(code_gen)
    disttriu_vec = distmat[np.triu_indices(40, 1)]
    angdist_col.append(disttriu_vec)
# pergen_std = np.array([.std(axis=0, ddof=1).mean(axis=0)
#                        for i in range(generations.max()+1)])
#%%
plt.plot([angdist.mean() for angdist in angdist_col])
plt.show()
#%%
plt.plot(np.arange(1, 76), codenorms**2)
plt.scatter(generations+1, codesnorms_all**2, 9)
plt.show()

#%%
def load_trajectory_normdata_cma(Droot, ):
    subdirs = os.listdir(Droot)
    meta_col = []
    codenorm_col = []
    codenorm_all_col = []
    genstd_col = []
    for unitdir in tqdm(subdirs):
        # 'resnet50_linf8_.Linearfc_009_ns0.5'
        toks = re.findall("(.*)_(\.[^_]*)_(\d*)(.*)", unitdir)
        assert len(toks) == 1, "cannot match pattern for %s" % unitdir
        toks = toks[0]
        netname = toks[0]
        layername = toks[1]
        channum = int(toks[2])
        if len(toks[3]) > 0:
            nstok = re.findall("_ns(.*)", toks[3])[0]
            noise_level = float(nstok)
        else:
            noise_level = 0.0
        npzpaths = glob(join(Droot, unitdir, "CholeskyCMAES_*.npz"))
        for fpath in npzpaths:
            if "PCA" in fpath:
                continue
            shortfn = os.path.split(fpath)[1]
            # CholeskyCMAES_alexnet_.classifier.Linear6_003_77176.npz
            toks = re.findall("CholeskyCMAES_(.*)_(\.[^_]*)_(\d*)_(\d\d\d\d\d).npz", shortfn)
            assert len(toks) == 1, "cannot match pattern for %s" % shortfn
            assert toks[0][0] == netname
            assert toks[0][1] == layername
            assert channum == int(toks[0][2])
            RND = int(toks[0][3])
            expmeta = EasyDict(netname=netname, layername=layername, channum=channum,
                               noise_level=noise_level, RND=RND,
                               expdir=unitdir)
            data = np.load(fpath)
            codes_all = data["codes_all"]
            generations = data["generations"]
            scores_all = data["scores_all"]
            cleanscores_all = data["cleanscores_all"]
            meancodes = np.array([codes_all[generations == i, :].mean(axis=0) for i in range(generations.max() + 1)])
            codenorms_mean = np.linalg.norm(meancodes, axis=1)
            codenorms_all = np.linalg.norm(codes_all, axis=1)
            pergen_std = np.array([codes_all[generations == i, :].std(axis=0, ddof=1).mean(axis=0)
                                   for i in range(generations.max() + 1)])
            meta_col.append(expmeta)
            codenorm_col.append(codenorms_mean)
            codenorm_all_col.append(codenorms_all)
            genstd_col.append(pergen_std)
    meta_df = pd.DataFrame(meta_col)
    return codenorm_col, codenorm_all_col, genstd_col, meta_df

dataroot = r"E:\Cluster_Backup\cma_optim_cmp"
codenorm_col, codenorm_all_col, genstd_col, meta_df = load_trajectory_normdata_cma(dataroot, )
#%%
def load_trajectory_normdata_cma_ctrl(Droot, select_optim=None):
    meta_col = []
    codenorm_col = []
    codenorm_all_col = []
    genstd_col = []
    subdirs = os.listdir(Droot)
    if select_optim is not None:
        subdirs = select_optim
    for optimnm in tqdm(subdirs):
        npzpaths = glob(join(Droot, optimnm, "noisectrl_%s_*.npz"%optimnm))
        for fpath in tqdm(npzpaths):
            if "PCA" in fpath:
                continue
            shortfn = os.path.split(fpath)[1]
            # CholeskyCMAES_alexnet_.classifier.Linear6_003_77176.npz
            try:
                toks = re.findall("noisectrl_%s_rep(\d\d\d\d\d).npz"%optimnm, shortfn)
                RND = int(toks[0][0])
                expmeta = EasyDict(RND=RND, expdir=optimnm)
                data = np.load(fpath)
                codes_all = data["codes_all"]
                generations = data["generations"]
                scores_all = data["scores_all"]
                # cleanscores_all = data["cleanscores_all"]
                meancodes = np.array([codes_all[generations == i, :].mean(axis=0) for i in range(generations.max() + 1)])
                codenorms_mean = np.linalg.norm(meancodes, axis=1)
                codenorms_all = np.linalg.norm(codes_all, axis=1)
                pergen_std = np.array([codes_all[generations == i, :].std(axis=0, ddof=1).mean(axis=0)
                                       for i in range(generations.max() + 1)])
                meta_col.append(expmeta)
                codenorm_col.append(codenorms_mean)
                codenorm_all_col.append(codenorms_all)
                genstd_col.append(pergen_std)
            except :
                continue
    meta_df = pd.DataFrame(meta_col)
    return codenorm_col, codenorm_all_col, genstd_col, meta_df

ctrlroot = r"F:\insilico_exps\noise_optim_ctrl"
codenorm_col_ctrl, codenorm_all_col_ctrl, genstd_col_ctrl, meta_df_ctrl = \
    load_trajectory_normdata_cma_ctrl(ctrlroot, select_optim=["CholeskyCMAES"])
#%%
plt.figure()
plt.plot(np.array(codenorm_col).T**2, color="blue",alpha=0.1)
plt.plot(np.array(codenorm_col_ctrl).T**2, color="red",alpha=0.1)
plt.show()
#%%
pkl.dump(EasyDict(codenorm_col=codenorm_col, codenorm_all_col=codenorm_all_col,
                  genstd_col=genstd_col, meta_df=meta_df, ),
         open("data\\evol_codenorm_std_data.pkl", 'wb'))
pkl.dump(EasyDict(codenorm_col_ctrl=codenorm_col_ctrl, codenorm_all_col_ctrl=codenorm_all_col_ctrl,
                  genstd_col_ctrl=genstd_col_ctrl, meta_df_ctrl=meta_df_ctrl, ),
         open("data\\ctrl_codenorm_std_data.pkl", 'wb'))
#%%
data = pkl.load(open("data\\evol_codenorm_std_data.pkl", 'rb'))
data_nois = pkl.load(open("data\\ctrl_codenorm_std_data.pkl", 'rb'))
codenorm_col = data.codenorm_col
codenorm_col_ctrl = data_nois.codenorm_col_ctrl
#%%
figdir = r"E:\OneDrive - Harvard University\GECCO2022\Figures\Norm_cmp"
plt.figure(figsize=[5,5])
genarr = np.arange(1,76)
plt.plot(genarr,np.array(codenorm_col).T**2, color="blue",alpha=0.1)
plt.plot(genarr,np.array(codenorm_col_ctrl).T**2, color="red",alpha=0.1)
plt.title("Norm of Mean Code During Evolution\n AlexNet vs Noise Driven Evolution\n"+statsummary,fontsize=16)
plt.legend(["Evolution (N=1050)","Noise Evol (N=100)"])
plt.ylabel("squared norm",fontsize=14)
plt.xlabel("Generations",fontsize=14)
plt.savefig(join(figdir, "noise_evol_sqnorm_cmp.png"))
plt.savefig(join(figdir, "noise_evol_sqnorm_cmp.pdf"))
plt.show()
#%%
plt.figure(figsize=[5,5])
genarr = np.arange(1,76)
plt.plot(genarr,np.array(codenorm_col).T, color="blue",alpha=0.1)
plt.plot(genarr,np.array(codenorm_col_ctrl).T, color="red",alpha=0.1)
plt.plot(genarr, np.sqrt((genarr-1)*4096))
plt.title("Norm of Mean Code During Evolution\n AlexNet vs Noise Driven Evolution\n"+statsummary,fontsize=16)
plt.legend(["Evolution (N=1050)","Noise Evol (N=100)"])
plt.ylabel("Norm",fontsize=14)
plt.xlabel("Generations",fontsize=14)
plt.savefig(join(figdir, "noise_evol_norm_cmp.png"))
plt.savefig(join(figdir, "noise_evol_norm_cmp.pdf"))
plt.show()
#%%
from scipy.stats import ranksums, ttest_ind
evol_finalnorm = np.array(codenorm_col)[:, -1]
ctrl_finalnorm = np.array(codenorm_col_ctrl)[:, -1]
tval,pval = ttest_ind(evol_finalnorm, ctrl_finalnorm)
rkval, rkpval = ranksums(evol_finalnorm, ctrl_finalnorm)
statsummary = f"Mean norm of evolution experiments {evol_finalnorm.mean():.2f}$\pm${evol_finalnorm.std():.2f} (mean $\pm$ std)"\
      f" in contrast to the control evolution {ctrl_finalnorm.mean():.2f}$\pm${ctrl_finalnorm.std():.2f}"\
      f" (t={tval:.3f} p={pval:.1e} df={len(evol_finalnorm)+len(ctrl_finalnorm)-2})"
#%%
print(f"Mean norm of evolution experiments {evol_finalnorm.mean():.2f}$\pm${evol_finalnorm.std():.2f} (mean $\pm$ std)"
      f" in contrast to the control evolution {ctrl_finalnorm.mean():.2f}$\pm${ctrl_finalnorm.std():.2f}"
      f" (t={tval:.3f} p={pval:.1e} df={len(evol_finalnorm)+len(ctrl_finalnorm)-2})")
#%%
R2_col = [np.corrcoef(cnorm**2, np.arange(1, 76))[0,1]**2 for cnorm in codenorm_col]
R2_col_ctrl = [np.corrcoef(cnorm**2, np.arange(1, 76))[0,1]**2 for cnorm in codenorm_col_ctrl]
print(f"Evolution {np.mean(R2_col):.4f}+-{np.std(R2_col):.4f} N={len(R2_col):d}\n"
      f"Control {np.mean(R2_col_ctrl):.4f}+-{np.std(R2_col_ctrl):.4f} N={len(R2_col_ctrl):d} ")

#%%
evol_finalstd = np.array(genstd_col)[:, -1]
ctrl_finalstd = np.array(genstd_col_ctrl)[:, -1]
tval,pval = ttest_ind(evol_finalstd, ctrl_finalstd)
rkval, rkpval = ranksums(evol_finalstd, ctrl_finalstd)

print(f"Mean std of evolution experiments {evol_finalstd.mean():.2f}$\pm${evol_finalstd.std():.2f}"
      f" in contrast to the control evolution {ctrl_finalstd.mean():.2f}$\pm${ctrl_finalstd.std():.2f}"
      f" (t={tval:.3f} p={pval:.1e} df={len(evol_finalstd)+len(ctrl_finalstd)-2})")


#%%
import torch
from lpips import LPIPS
G = upconvGAN("fc6")
G.cuda().eval().requires_grad_(False)
Dist = LPIPS(net="vgg", lpips=True, pretrained=True, )
Dist.cuda()
#%%
dist_col = []
for geni in tqdm(range(1, 75)):
    imgs = G.visualize_batch_np(codes_all[generations == geni,:])
    with torch.no_grad():
        dists = Dist(imgs[0].cuda(), imgs[1:].cuda()).cpu()
    dist_col.append(dists.numpy())
#%%
plt.plot([dists.mean() for dists in dist_col])
plt.show()
#%%
plt.plot(pergen_std)
plt.show()
#%%
pixdist_col = []
for geni in tqdm(range(1, 75)):
    imgs = G.visualize_batch_np(codes_all[generations == geni,:])
    with torch.no_grad():
        dists_pix = torch.pdist(imgs.view([40,-1]).cuda()).cpu()
    pixdist_col.append(dists_pix.numpy())
#%%


#%%
plt.plot([dists.mean() for dists in pixdist_col])
plt.show()
#%%

plt.figure()
plt.plot(genstd_col[200]/(500+codenorm_col[200]))
plt.show()
#%%
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
angdistmat = cosine_distances(meancodes[1:,:])
L2distmat = euclidean_distances(meancodes[1:,:])
#%%
plt.figure()
plt.subplot(121)
plt.matshow(angdistmat,fignum=0)
plt.subplot(122)
plt.matshow(L2distmat,fignum=0)
plt.show()
#%%
imgs_all = G.visualize_batch_np(meancodes)
#%%
import torchvision
import torch.nn.functional as F
from torchvision.transforms import ToPILImage,Resize,Normalize,Compose
from core.layer_hook_utils import featureFetcher
net = torchvision.models.alexnet(pretrained=True)
net.cuda().eval().requires_grad_(False)
#%%
featlayer = ".classifier.Linear1" # ".features.ReLU11"
feat = featureFetcher(net)
feat.record(featlayer)
#%%
RGBmean = torch.tensor([0.485, 0.456, 0.406]).view([1, 3, 1, 1]).cuda()
RGBstd = torch.tensor([0.229, 0.224, 0.225]).view([1, 3, 1, 1]).cuda()
imgpp = (F.interpolate(imgs_all.cuda(), [224, 224]) - RGBmean) / RGBstd
with torch.no_grad():
    net(imgpp)
feats_all = feat[featlayer].view([75, -1]).cpu()
#%%
angdistmat_feat = cosine_distances(feats_all[1:, :])
L2distmat_feat = euclidean_distances(feats_all[1:, :])
plt.figure()
plt.subplot(121)
plt.matshow(angdistmat_feat, fignum=0)
plt.subplot(122)
plt.matshow(L2distmat_feat, fignum=0)
plt.show()
#%%
angdistmat_img = cosine_distances(imgs_all.view(75,-1))
L2distmat_img = euclidean_distances(imgs_all.view(75,-1))
#%%
plt.matshow(L2distmat_img)
plt.show()
#%%
dmat = []
imgs_all = imgs_all.cpu()
for i in range(len(imgs_all)):
    with torch.no_grad():
        dvec = Dist(imgs_all[i],imgs_all[:]).cpu()
    dmat.append(dvec.squeeze().numpy())
#%%
plt.matshow(dmat)
plt.show()