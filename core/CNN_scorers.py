"""
pytorch CNN scorer that creates functions that map images to 
activation values of their hidden units. 
"""
import os
from os.path import join
from sys import platform
from time import time, sleep
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import torch
from torchvision import transforms
from torchvision import models
import torch.nn.functional as F
from core.GAN_utils import upconvGAN
from core.layer_hook_utils import layername_dict, register_hook_by_module_names, get_module_names, named_apply

if platform == "linux": # cluster
    # torchhome = "/scratch/binxu/torch/checkpoints"  # CHPC
    if "ris.wustl.edu" in os.environ['HOSTNAME']:
        scratchdir = os.environ["SCRATCH1"]
        torchhome = join(scratchdir, "torch/checkpoints")  # RIS
    else:
        torchhome = torch.hub._get_torch_home()
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        torchhome = r"E:\Cluster_Backup\torch"
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  ## Home_WorkStation
        torchhome = r"E:\Cluster_Backup\torch"
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-9LH02U9':  ## Home_WorkStation Victoria
        torchhome = r"E:\Cluster_Backup\torch"
    else:
        torchhome = torch.hub._get_torch_home()


class TorchScorer:
    """ Pure PyTorch CNN Scorer using hooks to fetch score from any layer in the net.
    Compatible with all models in torchvision zoo
    Demo:
        scorer = TorchScorer("vgg16")
        scorer.select_unit(("vgg16", "fc2", 10, 10, 10))
        scorer.score([np.random.rand(224, 224, 3), np.random.rand(227,227,3)])
        # if you want to record all the activation in conv1 layer you can use
        CNN.set_recording( 'conv2' ) # then CNN.artiphys = True
        scores, activations = CNN.score(imgs)

    """
    def __init__(self, model_name, imgpix=227, rawlayername=True, device="cuda"):
        self.imgpix = imgpix
        if isinstance(model_name, torch.nn.Module):
            self.model = model_name
            self.inputsize = (3, imgpix, imgpix)
            self.layername = None
        elif isinstance(model_name, str):
            if model_name == "vgg16":
                self.model = models.vgg16(pretrained=True)
                self.layers = list(self.model.features) + list(self.model.classifier)
                # self.layername = layername_dict[model_name]
                self.layername = None if rawlayername else layername_dict["vgg16"]
                self.inputsize = (3, imgpix, imgpix)
            elif model_name == "vgg16-face":
                self.model = models.vgg16(pretrained=False, num_classes=2622)
                self.model.load_state_dict(torch.load(join(torchhome, "vgg16_face.pt")))
                self.layers = list(self.model.features) + list(self.model.classifier)
                self.layername = None if rawlayername else layername_dict["vgg16"]
                self.inputsize = (3, imgpix, imgpix)
            elif model_name == "alexnet":
                self.model = models.alexnet(pretrained=True)
                self.layers = list(self.model.features) + list(self.model.classifier)
                self.layername = None if rawlayername else layername_dict[model_name]
                self.inputsize = (3, imgpix, imgpix)
            elif model_name == "densenet121":
                self.model = models.densenet121(pretrained=True)
                self.layers = list(self.model.features) + [self.model.classifier]
                self.layername = None if rawlayername else layername_dict[model_name]
                self.inputsize = (3, imgpix, imgpix)
            elif model_name == "densenet169":
                self.model = models.densenet169(pretrained=True)
                self.layername = None
                self.inputsize = (3, imgpix, imgpix)
            elif model_name == "resnet101":
                self.model = models.resnet101(pretrained=True)
                self.inputsize = (3, imgpix, imgpix)
                self.layername = None
            elif "resnet50" in model_name:
                if "resnet50-face" in model_name:  # resnet trained on vgg-face dataset.
                    self.model = models.resnet50(pretrained=False, num_classes=8631)
                    if model_name == "resnet50-face_ft":
                        self.model.load_state_dict(torch.load(join(torchhome, "resnet50_ft_weight.pt")))
                    elif model_name == "resnet50-face_scratch":
                        self.model.load_state_dict(torch.load(join(torchhome, "resnet50_scratch_weight.pt")))
                    else:
                        raise NotImplementedError("Feasible names are resnet50-face_scratch, resnet50-face_ft")
                else:
                    self.model = models.resnet50(pretrained=True)
                    if model_name == "resnet50_linf_8":  # robust version of resnet50.
                        self.model.load_state_dict(torch.load(join(torchhome, "imagenet_linf_8_pure.pt")))
                    elif model_name == "resnet50_linf_4":
                        self.model.load_state_dict(torch.load(join(torchhome, "imagenet_linf_4_pure.pt")))
                    elif model_name == "resnet50_l2_3_0":
                        self.model.load_state_dict(torch.load(join(torchhome, "imagenet_l2_3_0_pure.pt")))
                self.inputsize = (3, imgpix, imgpix)
                self.layername = None
            elif model_name == "cornet_s":
                from cornet import cornet_s
                Cnet = cornet_s(pretrained=True)
                self.model = Cnet.module
                self.inputsize = (3, imgpix, imgpix)
                self.layername = None
            else:
                raise NotImplementedError("Cannot find the specified model %s"%model_name)
        else:
            raise NotImplementedError("model_name need to be either string or nn.Module")

        self.model.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        # self.preprocess = transforms.Compose([transforms.ToPILImage(),
        #                                       transforms.Resize(size=(224, 224)),
        #                                       transforms.ToTensor(),
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])  # Imagenet normalization RGB
        self.RGBmean = torch.tensor([0.485, 0.456, 0.406]).view([1, 3, 1, 1]).to(device)
        self.RGBstd = torch.tensor([0.229, 0.224, 0.225]).view([1, 3, 1, 1]).to(device)
        self.device = device
        self.hooks = []
        self.artiphys = False
        self.record_layers = []
        self.recordings = {}

        self.activation = {}

    def get_activation(self, name, unit=None, unitmask=None, ingraph=False):
        """
        :parameter
            name: key to retrieve the recorded activation `self.activation[name]`
            unit: a tuple of 3 element or single element. (chan, i, j ) or (chan)
            unitmask: used in population recording, it could be a binary mask of the same shape / length as the
                element number in feature tensor. Or it can be an array of integers.

            *Note*: when unit and unitmask are both None, this function will record the whole output feature tensor.

            ingraph: if True, then the recorded activation is still connected to input, so can pass grad.
                    if False then cannot
        :return
            hook:  Return a hook function that record the unit activity into the entry in activation dict of scorer.
        """
        if unit is None and unitmask is None:  # if no unit is given, output the full tensor. 
            def hook(model, input, output): 
                self.activation[name] = output.clone() if ingraph else output.clone().detach()

        elif unitmask is not None:
            # has a unit mask, which could be an index list or a tensor mask same shape of the 3 dimensions.
            def hook(model, input, output): 
                out = output.clone() if ingraph else output.clone().detach()
                Bsize = out.shape[0]
                self.activation[name] = out.view([Bsize, -1])[:, unitmask.reshape(-1)]

        else:
            def hook(model, input, output):
                #   Add clone so that future inplace relu does not overwrite nonrelu unit scores.
                out = output.clone() if ingraph else output.clone().detach()
                if len(output.shape) == 4: 
                    self.activation[name] = out[:, unit[0], unit[1], unit[2]]
                elif len(output.shape) == 2: 
                    self.activation[name] = out[:, unit[0]]

        return hook

    def set_unit(self, reckey, layer, unit=None, ingraph=False):
        if self.layername is not None:
            # if the network is a single stream feedforward structure, we can index it and use it to find the
            # activation
            idx = self.layername.index(layer)
            handle = self.layers[idx].register_forward_hook(self.get_activation(reckey, unit, ingraph=ingraph)) # we can get the layer by indexing
            self.hooks.append(handle)  # save the hooks in case we will remove it.
        else:
            # if not, we need to parse the architecture of the network.
            # indexing is not available, we need to register by recursively visit the layers and find match.
            handle, modulelist, moduletype = register_hook_by_module_names(layer, self.get_activation(reckey, unit, ingraph=ingraph),
                                self.model, self.inputsize, device="cuda")
            self.hooks.extend(handle)  # handle here is a list.
        return handle

    def set_units_by_mask(self, reckey, layer, unit_mask=None):
        if self.layername is not None:
            # if the network is a single stream feedforward structure, we can index it and use it to find the
            # activation
            idx = self.layername.index(layer)
            handle = self.layers[idx].register_forward_hook(self.get_activation(reckey, unitmask=unit_mask))
            # we can get the layer by indexing
            self.hooks.append(handle)  # save the hooks in case we will remove it.
        else:
            # if not, we need to parse the architecture of the network indexing is not available. 
            # we need to register by recursively visit the layers and find match.
            handle, modulelist, moduletype = register_hook_by_module_names(layer, 
                self.get_activation(reckey, unitmask=unit_mask), self.model, self.inputsize, device="cuda")
            self.hooks.extend(handle)  # handle here is a list.
        return handle
    
    def select_unit(self, unit_tuple, allow_grad=False):
        # self._classifier_name = str(unit_tuple[0])
        self.layer = str(unit_tuple[1])
        # `self._net_layer` is used to determine which layer to stop forwarding
        self.chan = int(unit_tuple[2])
        if len(unit_tuple) == 5:
            self.unit_x = int(unit_tuple[3])
            self.unit_y = int(unit_tuple[4])
        else:
            self.unit_x = None
            self.unit_y = None
        self.set_unit("score", self.layer, unit=(self.chan, self.unit_x, self.unit_y), ingraph=allow_grad)

    def set_recording(self, record_layers):
        """The function to select a scalar output from a NN"""
        self.artiphys = True  # flag to record the neural activity in one layer
        self.record_layers.extend(record_layers)
        for layer in record_layers:  # will be arranged in a dict of lists
            self.set_unit(layer, layer, unit=None)
            self.recordings[layer] = []

    def set_popul_recording(self, record_layer, mask):
        self.artiphys = True
        self.record_layers.append(record_layer)
        h = self.set_units_by_mask(record_layer, record_layer, unit_mask=mask)
        self.recordings[record_layer] = []

    def preprocess(self, img, input_scale=255):
        """preprocess single image array or a list (minibatch) of images
        This includes Normalize using RGB mean and std and resize image to (227, 227)
        """
        # could be modified to support batch processing. Added batch @ July. 10, 2020
        # test and optimize the performance by permute the operators. Use CUDA acceleration from preprocessing
        if type(img) is list: # the following lines have been optimized for speed locally.
            img_tsr = torch.stack(tuple(torch.from_numpy(im) for im in img)).to(self.device).float().permute(0, 3, 1, 2) / input_scale
            img_tsr = (img_tsr - self.RGBmean) / self.RGBstd
            resz_out_tsr = F.interpolate(img_tsr, (self.imgpix, self.imgpix), mode='bilinear',
                                         align_corners=True)
            return resz_out_tsr
        elif type(img) is torch.Tensor:
            img_tsr = (img.to(self.device) / input_scale - self.RGBmean) / self.RGBstd
            resz_out_tsr = F.interpolate(img_tsr, (self.imgpix, self.imgpix), mode='bilinear',
                                         align_corners=True)
            return resz_out_tsr
        elif type(img) is np.ndarray and img.ndim == 4:
            img_tsr = torch.tensor(img / input_scale).float().permute(0,3,1,2).to(self.device)
            img_tsr = (img_tsr - self.RGBmean) / self.RGBstd
            resz_out_tsr = F.interpolate(img_tsr, (self.imgpix, self.imgpix), mode='bilinear',
                                         align_corners=True)
            return resz_out_tsr
        elif type(img) is np.ndarray and img.ndim in [2, 3]:  # assume it's individual image
            img_tsr = transforms.ToTensor()(img / input_scale).float()
            img_tsr = self.normalize(img_tsr).unsqueeze(0)
            resz_out_img = F.interpolate(img_tsr, (self.imgpix, self.imgpix), mode='bilinear',
                                         align_corners=True)
            return resz_out_img
        else:
            raise ValueError

    def score(self, images, with_grad=False, B=42, input_scale=1.0):
        """Score in batch will accelerate processing greatly! """ # assume image is using 255 range
        scores = np.zeros(len(images))
        csr = 0  # if really want efficiency, we should use minibatch processing.
        imgn = len(images)
        for layer in self.recordings: 
            self.recordings[layer] = []
        while csr < imgn:
            csr_end = min(csr + B, imgn)
            img_batch = self.preprocess(images[csr:csr_end], input_scale=input_scale)
            with torch.no_grad():
                self.model(img_batch)
            if "score" in self.activation: # if score is not there set trace to zero. 
                scores[csr:csr_end] = self.activation["score"].squeeze().cpu().numpy().squeeze()

            if self.artiphys:  # record the whole layer's activation
                for layer in self.record_layers:
                    score_full = self.activation[layer] # temporory storage
                    self.recordings[layer].append(score_full.cpu().numpy()) # formulated storage
            
            csr = csr_end
        
        for layer in self.recordings: 
            self.recordings[layer] = np.concatenate(self.recordings[layer],axis=0)

        if self.artiphys:
            return scores, self.recordings
        else:
            return scores

    def score_tsr(self, img_tsr, with_grad=False, B=42, input_scale=1.0):
        """Score in batch will accelerate processing greatly!
        img_tsr is already torch.Tensor
        """
        # assume image is using 255 range
        imgn = img_tsr.shape[0]
        scores = np.zeros(img_tsr.shape[0])
        for layer in self.recordings: 
            self.recordings[layer] = []
        csr = 0  # if really want efficiency, we should use minibatch processing.
        while csr < imgn:
            csr_end = min(csr + B, imgn)
            img_batch = self.preprocess(img_tsr[csr:csr_end,:,:,:], input_scale=input_scale)
            with torch.no_grad():
                self.model(img_batch.to(self.device))
            if "score" in self.activation: # if score is not there set trace to zero. 
                scores[csr:csr_end] = self.activation["score"].squeeze().cpu().numpy().squeeze()

            if self.artiphys:  # record the whole layer's activation
                for layer in self.record_layers:
                    score_full = self.activation[layer]
                    self.recordings[layer].append(score_full.cpu().numpy())

            csr = csr_end

        for layer in self.recordings: 
            self.recordings[layer] = np.concatenate(self.recordings[layer],axis=0)

        if self.artiphys:
            return scores, self.recordings
        else:
            return scores

    def score_tsr_wgrad(self, img_tsr, B=10, input_scale=1.0):
        imgn = img_tsr.shape[0]
        scores = torch.zeros(img_tsr.shape[0]).to(self.device)
        csr = 0  # if really want efficiency, we should use minibatch processing.
        while csr < imgn:
            csr_end = min(csr + B, imgn)
            img_batch = self.preprocess(img_tsr[csr:csr_end,:,:,:], input_scale=input_scale)
            self.model(img_batch)
            if "score" in self.activation:  # if score is not there set trace to zero.
                scores[csr:csr_end] += self.activation["score"].squeeze()
            if self.artiphys:  # record the whole neurlayer's activation
                for layer in self.record_layers:
                    score_full = self.activation[layer]
                    self.recordings[layer].append(score_full.cpu().numpy())

            csr = csr_end

        if self.artiphys:
            return scores, self.recordings
        else:
            return scores

