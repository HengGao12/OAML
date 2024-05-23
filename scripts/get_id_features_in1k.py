import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
from timm.models.vision_transformer import VisionTransformer  


config_files = [
    './configs/datasets/imagenet/imagenet.yml',
    './configs/datasets/imagenet/imagenet_ood.yml',
    # './configs/networks/resnet18_224x224.yml',
    './configs/networks/resnet50.yml',
    './configs/pipelines/test/test_ood.yml',
    './configs/preprocessors/base_preprocessor.yml',
    './configs/postprocessors/ebo.yml',   
]
config = config.Config(*config_files)

# config.network.checkpoint = 'results/pretrained_weights/resnet50_imagenet1k_v1.pth'

# vit_cifar10 /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/pytorch_model.bin
# res18_224x224 /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_base_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_e500/s0/best_epoch469_acc0.9550.ckpt

config.network.pretrained = False
config.num_workers = 8
config.save_output = False
config.parse_refs()

setup_logger(config)

# net = VisionTransformer(num_classes=1000).cuda()
# net.load_state_dict(
#     torch.load('/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/vit_imagenet_pretrained.bin')
# )
net = get_network(config.network)
net.cuda()
net.load_state_dict(
    torch.load('results/pretrained_weights/resnet50_imagenet1k_v1.pth')
)

id_loader_dict = get_dataloader(config)
ood_loader_dict = get_ood_dataloader(config)
num_classes = 1000
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
data_dict = torch.zeros(num_classes, 500, 2048).cuda()
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
        target_numpy = target.cpu().data.numpy()
        for index in range(len(target)):
            dict_key = target_numpy[index]
            if number_dict[dict_key] < 500:
                data_dict[dict_key][number_dict[dict_key]] = feat[index].detach()
                number_dict[dict_key] += 1


print(data_dict.shape)
np.save('/home1/gaoheng/gh_workspace/OAML/id_feat_in1k_res50.npy', data_dict.cpu().numpy())