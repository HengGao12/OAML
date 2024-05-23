import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import sys
sys.path.append('/home1/gaoheng/gh_workspace/OAML')
from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_224x224
from openood.evaluators.ood_evaluator import OODEvaluator
from openood.utils import config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger
import time
import timm
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import ipdb


config_files = [
    './configs/datasets/cifar10/cifar10_224x224.yml',
    './configs/datasets/cifar10/cifar10_ood.yml',
    './configs/networks/resnet18_224x224.yml',
    # './configs/networks/vit.yml',
    './configs/pipelines/test/test_ood.yml',
    './configs/preprocessors/base_preprocessor.yml',
    './configs/postprocessors/ebo.yml',   
]
config = config.Config(*config_files)

# config.network.checkpoint = '/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/pytorch_model.bin'

# vit_cifar10 /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/pytorch_model.bin
# res18_224x224 /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_base_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_e500/s0/best_epoch469_acc0.9550.ckpt

config.network.pretrained = True
config.num_workers = 8
config.save_output = False
config.parse_refs()

setup_logger(config)

# net = timm.create_model("timm/vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=False)
# net.head = nn.Linear(net.head.in_features, 10)
# Please use this 
net = get_network(config.network)
net.cuda()
net.load_state_dict(
    torch.load('/home1/gaoheng/gh_workspace/OAML/results/cifar10_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt')
)
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/pytorch_model.bin
# /home1/gaoheng/gh_workspace/OAML/results/cifar10_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt
id_loader_dict = get_dataloader(config)
ood_loader_dict = get_ood_dataloader(config)
num_classes = 10
modes = ['train'
        # 'test'
        #  ,'val'
        ]
dl = id_loader_dict['train']
dataiter = iter(dl)
# save ID features.
number_dict = {}
for i in range(num_classes):
    number_dict[i] = 0
net.eval()
# data_dict = torch.zeros(num_classes, 500, 768).cuda()
data_dict = torch.zeros(num_classes, 500, 512).cuda()
with torch.no_grad():
    for i in tqdm(range(1, 
                    len(dataiter)+1),
                    desc='Extracting reults...',
                    position=0,
                    leave=True):
        batch = next(dataiter)
        data = batch['data'].cuda()
        target = batch['label'].cuda()
        # data, target = data.cuda(), target.cuda()
        # forward
        logits, feat = net(data, return_feature=True)
        # ipdb.set_trace()
        target_numpy = target.cpu().data.numpy()
        for index in range(len(target)):
            dict_key = target_numpy[index]
            if number_dict[dict_key] < 500:
                data_dict[dict_key][number_dict[dict_key]] = feat[index].detach()
                number_dict[dict_key] += 1

np.save('/home1/gaoheng/gh_workspace/OAML/id_feat_cifar10_res18.npy', data_dict.cpu().numpy())