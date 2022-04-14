import os
from os.path import join
from sys import platform
from time import time
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import torch
import torch.nn.functional as F
from core.CNN_scorers import TorchScorer
from core.GAN_utils import upconvGAN
from core.Optimizers import CholeskyCMAES # HessAware_Gauss_DC,
default_init_sigma = 3.0
default_Aupdate_freq = 10
class ExperimentEvolution:
    def __init__(self, model_unit, max_step=100, imgsize=(227, 227), corner=(0, 0), optimizer=None,
                 savedir="", explabel="", GAN="fc6", device="cuda"):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        self.pref_unit = model_unit
        # AlexNet, VGG, ResNet, DENSE and anything else
        self.CNNmodel = TorchScorer(model_unit[0], device=device)
        self.CNNmodel.select_unit(model_unit)
        # Allow them to choose from multiple optimizers, substitute generator.visualize and render
        if GAN in ["fc6", "fc7", "fc8"]:
            self.G = upconvGAN(name=GAN).cuda()
            self.render_tsr = self.G.visualize_batch_np  # this output tensor
            self.render = self.G.render
            self.code_length = self.G.codelen  # 1000 "fc8" 4096 "fc6", "fc7"
        elif GAN == "BigGAN":
            from BigGAN_Evolution import BigGAN_embed_render
            self.render = BigGAN_embed_render
            self.code_length = 256  # 128 # 128d Class Embedding code or 256d full code could be used.
        else:
            raise NotImplementedError
        if optimizer is None:
            self.optimizer = CholeskyCMAES(self.code_length, population_size=None, init_sigma=default_init_sigma,
                                       init_code=np.zeros([1, self.code_length]), Aupdate_freq=default_Aupdate_freq,
                                       maximize=True, random_seed=None, optim_params={})
        else:
            self.optimizer = optimizer

        self.max_steps = max_step
        self.corner = corner  # up left corner of the image
        self.imgsize = imgsize  # size of image, allowing showing CNN resized image
        self.savedir = savedir
        self.explabel = explabel
        self.Perturb_vec = []

    def run(self, init_code=None):
        """Same as Resized Evolution experiment"""
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        for self.istep in range(self.max_steps):
            if self.istep == 0:
                if init_code is None:
                    codes = np.zeros([1, self.code_length])
                else:
                    codes = init_code
            t0 = time()
            self.current_images = self.render_tsr(codes)
            t1 = time()  # generate image from code
            self.current_images = resize_and_pad_tsr(self.current_images, self.imgsize, self.corner)
            # Fixed Jan.14 2021
            synscores = self.CNNmodel.score_tsr(self.current_images)
            t2 = time()  # score images
            codes_new = self.optimizer.step_simple(synscores, codes)
            t3 = time()  # use results to update optimizer
            self.codes_all.append(codes)
            self.scores_all = self.scores_all + list(synscores)
            self.generations = self.generations + [self.istep] * len(synscores)
            codes = codes_new
            # summarize scores & delays
            print('synth img scores: mean {:.3f} +- std {:.3f}'.format(np.nanmean(synscores), np.nanstd(synscores)))
            print(('step %d  time: total %.2fs | ' +
                   'GAN visualize %.2fs   CNN score %.2fs   optimizer step %.2fs')
                  % (self.istep, t3 - t0, t1 - t0, t2 - t1, t3 - t2))
        self.codes_all = np.concatenate(tuple(self.codes_all), axis=0)
        self.scores_all = np.array(self.scores_all)
        self.generations = np.array(self.generations)

    def save_last_gen(self, filename=""):
        idx = np.argmax(self.scores_all)
        select_code = self.codes_all[idx:idx + 1, :]
        lastgen_code = np.mean(self.codes_all[self.generations == max(self.generations), :], axis=0, keepdims=True)
        lastgen_score = np.mean(self.scores_all[self.generations == max(self.generations)], )
        np.savez(join(self.savedir, "Evolution_codes_%s.npz" % (self.explabel)),
                 best_code=select_code, best_score=self.scores_all[idx],
                 lastgen_codes=lastgen_code, lastgen_score=lastgen_score)
        print("Last generation and Best code saved.")

    def load_traj(self, filename):
        data = np.load(join(self.savedir, filename))
        self.codes_all = data["codes_all"]
        self.scores_all = data["scores_all"]
        self.generations = data["generations"]

    def analyze_traj(self):
        '''Get the trajectory and the PCs and the structures of it'''
        final_gen_norms = np.linalg.norm(self.codes_all[self.generations == max(self.generations), :], axis=1)
        self.sphere_norm = final_gen_norms.mean()
        code_pca = PCA(n_components=50)
        PC_Proj_codes = code_pca.fit_transform(self.codes_all)
        self.PC_vectors = code_pca.components_
        if PC_Proj_codes[-1, 0] < 0:  # decide which is the positive direction for PC1
            # this is important or the images we show will land in the opposite side of the globe.
            inv_PC1 = True
            self.PC_vectors[0, :] = - self.PC_vectors[0, :]
            self.PC1_sign = -1
        else:
            inv_PC1 = False
            self.PC1_sign = 1
            pass

    def visualize_best(self, show=False):
        idx = np.argmax(self.scores_all)
        select_code = self.codes_all[idx:idx+1, :]
        score_select = self.scores_all[idx]
        img_select = self.render(select_code, scale=1.0) #, scale=1
        fig = plt.figure(figsize=[3, 1.7])
        plt.subplot(1, 2, 1)
        plt.imshow(img_select[0])
        plt.axis('off')
        plt.title("{0:.2f}".format(score_select), fontsize=16)
        plt.subplot(1, 2, 2)
        resize_select = resize_and_pad(img_select, self.imgsize, self.corner, scale=1.0)
        plt.imshow(resize_select[0])
        plt.axis('off')
        plt.title("{0:.2f}".format(score_select), fontsize=16)
        if show:
            plt.show()
        fig.savefig(join(self.savedir, "Best_Img_%s.png" % (self.explabel)))
        return fig

    def visualize_trajectory(self, show=True):
        gen_slice = np.arange(min(self.generations), max(self.generations)+1)
        AvgScore = np.zeros_like(gen_slice).astype("float64")
        MaxScore = np.zeros_like(gen_slice).astype("float64")
        for i, geni in enumerate(gen_slice):
            AvgScore[i] = np.mean(self.scores_all[self.generations == geni])
            MaxScore[i] = np.max(self.scores_all[self.generations == geni])
        figh = plt.figure()
        plt.scatter(self.generations, self.scores_all, s=16, alpha=0.6, label="all score")
        plt.plot(gen_slice, AvgScore, color='black', label="Average score")
        plt.plot(gen_slice, MaxScore, color='red', label="Max score")
        plt.xlabel("generation #")
        plt.ylabel("CNN unit score")
        plt.title("Optimization Trajectory of Score\n")# + title_str)
        plt.legend()
        if show:
            plt.show()
        figh.savefig(join(self.savedir, "Evolv_Traj_%s.png" % (self.explabel)))
        return figh


from skimage.transform import rescale, resize
def resize_and_pad_tsr(img_tsr, size, offset, canvas_size=(227, 227), scale=1.0):
    '''Resize and Pad a list of images to list of images
    Note this function is assuming the image is in (0,1) scale so padding with 0.5 as gray background.
    '''
    assert img_tsr.ndim in [3, 4]
    if img_tsr.ndim == 3:
        img_tsr.unsqueeze_(0)
    imgn = img_tsr.shape[0]

    padded_shape = (imgn, 3,) + canvas_size
    pad_img = torch.ones(padded_shape) * 0.5 * scale
    pad_img.to(img_tsr.dtype)
    rsz_tsr = F.interpolate(img_tsr, size=size)
    pad_img[:, :, offset[0]:offset[0] + size[0], offset[1]:offset[1] + size[1]] = rsz_tsr
    return pad_img


def resize_and_pad(img_list, size, offset, canvas_size=(227, 227), scale=1.0):
    '''Resize and Pad a list of images to list of images
    Note this function is assuming the image is in (0,1) scale so padding with 0.5 as gray background.
    '''
    resize_img = []
    padded_shape = canvas_size + (3,)
    for img in img_list:
        if img.shape == padded_shape:  # save some computation...
            resize_img.append(img.copy())
        else:
            pad_img = np.ones(padded_shape) * 0.5 * scale
            pad_img[offset[0]:offset[0]+size[0], offset[1]:offset[1]+size[1], :] = resize(img, size, )#cv2.INTER_AREA)
            resize_img.append(pad_img.copy())
    return resize_img

if __name__ == "__main__":
    explabel, model_unit = "alexnet_fc8_1", ("alexnet", ".classifier.Linear6", 1)
    explabel, model_unit = "vgg16_fc8_1", ("vgg16", ".classifier.Linear6", 1)
    # explabel, model_unit = "densenet_fc1", ("densenet121", ".Linearclassifier", 1)
    Exp = ExperimentEvolution(model_unit, savedir=r"E:\Cluster_Backup\Evol_tmp", explabel=explabel, )
    Exp.run()
    Exp.visualize_best()
    Exp.visualize_trajectory()
    Exp.save_last_gen()