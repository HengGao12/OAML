import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
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
import ipdb

config_files = [
    './configs/datasets/cifar10/cifar10_224x224.yml',
    './configs/datasets/cifar10/cifar10_ood.yml',
    './configs/networks/resnet18_224x224.yml',
    # './configs/networks/vit.yml',
    './configs/pipelines/test/test_ood.yml',
    './configs/preprocessors/base_preprocessor.yml',
    './configs/postprocessors/otdp.yml',
]
config = config.Config(*config_files)

config.network.checkpoint = '/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_base_generative_ood_distill_trainer_e500_lr0.001_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_0.2dcl_0.001mi_loss_0.001ml_0.001dml/s0/best_epoch387_acc0.9530.ckpt'
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_base_generative_ood_distill_trainer_e500_lr0.001_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_0.2dcl_0.001mi_loss_0.001ml_0.001dml/s0/best_epoch387_acc0.9530.ckpt
# /public/home/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt
# vit_cifar10 /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/pytorch_model.bin
# res18_224x224 /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_base_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_e500/s0/best_epoch469_acc0.9550.ckpt
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt
# /public/home/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_base_generative_ood_distill_trainer_e500_lr0.001_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_0.2dcl_0.001mi_loss_0.001ml_0.001dml/s0/best_epoch387_acc0.9530.ckpt
config.network.pretrained = True
config.num_workers = 8
config.save_output = False
config.parse_refs()

setup_logger(config)

# net = timm.create_model("timm/vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=False)
# net.head = nn.Linear(net.head.in_features, 10)
# Please use this 
net = get_network(config.network)

# net.cuda()
# net.load_state_dict(
#     torch.load('/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/vit_cifar10_finetuned.bin'), strict=False
# )
# net.cuda()
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/pytorch_model.bin
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt
net.eval()
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar10_resnet18_224x224_base_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_e500/s0/best_epoch469_acc0.9550.ckpt

id_loader_dict = get_dataloader(config)
ood_loader_dict = get_ood_dataloader(config)

evaluator = get_evaluator(config)
postprocessor = get_postprocessor(config)
postprocessor.setup(net, id_loader_dict, ood_loader_dict)
# postprocessor_name = 'vim'
print('\n', flush=True)
print(u'\u2500' * 70, flush=True)

# start calculating accuracy
print('\nStart evaluation...', flush=True)



# evaluator = Evaluator(
#     net,
#     id_name='cifar10',                     # the target ID dataset
#     data_root='./data',                    # change if necessary
#     config_root=None,                      # see notes above
#     preprocessor=None,                     # default preprocessing for the target ID dataset
#     postprocessor_name=postprocessor_name, # the postprocessor to use
#     # postprocessor=None,                    # if you want to use your own postprocessor
#     batch_size=200,                        # for certain methods the results can be slightly affected by batch size
#     shuffle=False,
#     num_workers=0)  

acc_metrics = evaluator.eval_acc(net, id_loader_dict['test'],
                                    postprocessor)
# print(acc)
print('\nAccuracy {:.2f}%'.format(100 * acc_metrics['acc']),
        flush=True)
print(u'\u2500' * 70, flush=True)

timer = time.time()
evaluator.eval_ood(net, id_loader_dict, ood_loader_dict, postprocessor)
print('Time used for eval_ood: {:.0f}s'.format(time.time() - timer))
print('Completed!', flush=True)

# print('Componets within evaluator')
# print('The OOD score of the first 5 samples of CIFAR-100:\t', evaluator.scores['ood']['near']['cifar100'][1][:5])

