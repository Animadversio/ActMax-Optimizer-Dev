from core.GAN_utils import upconvGAN
import torch
import numpy as np
G = upconvGAN("fc6")
G.requires_grad_(False)
#%%
img = G.visualize(torch.randn(1, 4096))
#%%

# perturb_vec = perturb_vec/perturb_vec.norm(dim=1, keepdim=True)
DeltaNorm = 5
#%%
from os.path import join
from core.montage_utils import make_grid, make_grid_np, ToPILImage, ToTensor
def normalize(vecs, R=1):
    return R * vecs / vecs.norm(dim=1,keepdim=True)
figdir = r"E:\OneDrive - Harvard University\GECCO2022\Figures\GANsphere_geom"
#%%
z0 = torch.randn(1, 4096)
perturb_vec = torch.randn(1, 4096)
unitz = z0 / z0.norm()
tang_vec = perturb_vec - perturb_vec @ unitz.T @ unitz
# tang_vec = tang_vec/tang_vec.norm(dim=1, keepdim=True)
tang_id_vec = torch.cat((-tang_vec,torch.zeros(1,4096),tang_vec),dim=0)
#%%
imgs_5norm = G.visualize(5*z0+tang_id_vec)
imgs_3norm = G.visualize(3*z0+tang_id_vec)
imgs_1norm = G.visualize(1*z0+tang_id_vec)
mtg1 = ToPILImage()(make_grid(torch.cat((imgs_1norm,imgs_3norm,imgs_5norm),dim=0),nrow=3))
mtg1.show()
mtg1.save(join(figdir,"GAN_geom_perturb_montage_eucl.png"))
#%%
imgs_5norm = G.visualize(normalize(5*z0+5*tang_id_vec, (5*z0).norm()))
imgs_3norm = G.visualize(normalize(3*z0+3*tang_id_vec, (3*z0).norm()))
imgs_1norm = G.visualize(normalize(1*z0+tang_id_vec, (1*z0).norm()))
mtg2 = ToPILImage()(make_grid(torch.cat((imgs_1norm,imgs_3norm,imgs_5norm),dim=0),nrow=3))
mtg2.show()
mtg2.save(join(figdir,"GAN_geom_perturb_montage_angl.png"))
#%%
#%%
# imgs_5norm = G.visualize(5*z0+5*tang_id_vec)
# imgs_3norm = G.visualize(3*z0+3*tang_id_vec)
# imgs_1norm = G.visualize(1*z0+tang_id_vec)
# ToPILImage()(make_grid(torch.cat((imgs_1norm, imgs_3norm, imgs_5norm),dim=0),nrow=3)).show()
imgs_scales = G.visualize()
#%%
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
angdistmat = cosine_distances(imgs_scales.view(5,-1))
#%%
def show_tsrbatch(imgtsr, nrow=None):
    ToPILImage()(make_grid(imgtsr.cpu(), nrow=nrow)).show()

def PIL_tsrbatch(imgtsr, nrow=None):
    mtg = ToPILImage()(make_grid(imgtsr.cpu(), nrow=nrow))
    return mtg
#%%
show_tsrbatch(imgs_scales)
#%%
z0 = torch.randn(1, 4096)
pertbvec = 3*torch.randn(1, 4096)
unitz = z0 / z0.norm()
tang_vec = pertbvec - pertbvec @ unitz.T @ unitz
code_scales = torch.arange(1, 6).float().unsqueeze(1)@z0
imgtsrs = torch.cat((G.visualize(code_scales),
    G.visualize(code_scales + tang_vec),
    G.visualize(normalize(code_scales + torch.arange(1.0, 6.0).unsqueeze(1)@tang_vec, code_scales.norm(dim=1,keepdim=True)))), dim=0)
show_tsrbatch(imgtsrs, nrow=5)
