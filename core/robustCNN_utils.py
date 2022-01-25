import torch
import torchvision
from os.path import join
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
ckpt_root = r"D:\DL_Projects\Vision\AdvPretrained_models"
pytorch_models = {
        'alexnet': torchvision.models.alexnet,
        'vgg16': torchvision.models.vgg16,
        'vgg16_bn': torchvision.models.vgg16_bn,
        'squeezenet': torchvision.models.squeezenet1_0,
        'densenet': torchvision.models.densenet161,
        'shufflenet': torchvision.models.shufflenet_v2_x1_0,
        'mobilenet': torchvision.models.mobilenet_v2,
        'resnet18' : torchvision.models.resnet18,
        'resnet50' : torchvision.models.resnet50,
        'wide_resnet50_2' : torchvision.models.wide_resnet50_2,
        'resnext50_32x4d': torchvision.models.resnext50_32x4d,
        'mnasnet': torchvision.models.mnasnet1_0,
    }

ckptpaths = {
     'densenet' : 'densenet_l2_eps3.ckpt',
     'mnasnet' : 'mnasnet_l2_eps3.ckpt',
     'mobilenet' : 'mobilenet_l2_eps3.ckpt',
     'resnet18' : 'resnet18_l2_eps3.ckpt',
     'resnet50' : 'resnet50_l2_eps3.ckpt',
     'resnext50_32x4d' : 'resnext50_32x4d_l2_eps3.ckpt',
     'wide_resnet50_2' : 'wide_resnet50_2_l2_eps3.ckpt',
     'wide_resnet50_4' : 'wide_resnet50_4_l2_eps3.ckpt',
     'shufflenet' : 'shufflenet_l2_eps3.ckpt',
     'vgg16_bn' : 'vgg16_bn_l2_eps3.ckpt',
 }


def load_pretrained_robust_model(arch, ckptname=None):
    constructor = pytorch_models[arch]
    if ckptname is None:
        ckptname = ckptpaths[arch]

    model, _ = make_and_restore_model(arch=constructor(pretrained=False),
              dataset=ImageNet("."), add_custom_forward=True,
              resume_path=join(ckpt_root, ckptname))
    return model.model.model
