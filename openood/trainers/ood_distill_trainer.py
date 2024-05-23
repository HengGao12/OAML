import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .lr_scheduler import cosine_annealing

import timm

from ..networks.kd_model import DT
from timm.models.vision_transformer import VisionTransformer  
from openood.losses.contrast_loss import CRDLoss
from openood.losses.mi_loss import CLUB
# from transformers import ViTModel
from openood.networks.knn import generate_outliers
import faiss
import ipdb
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
res = faiss.StandardGpuResources()
gpu_id=0
flat_config = faiss.GpuIndexFlatConfig()
flat_config.device = 0
KNN_index = faiss.GpuIndexFlatL2(res, 768, flat_config)
from torch.distributions import MultivariateNormal

sum_temp = 0
feature_map_inputs = []
feature_map_outputs = []


class OODDistillTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader, train_loader2: DataLoader,
                 config: Config, fea_dist=True, logits_distill=True, ood_fea_distill=True, dataset='cifar10') -> None:

        self.net = net
        self.train_loader = train_loader
        self.train_loader2 = train_loader2
        self.config = config
        self.logits_distill = logits_distill
        self.fea_distill = fea_dist
        self.ood_fea_distill = ood_fea_distill
        self.dataset = dataset
        if self.dataset == 'cifar10':
            self.num_classes = 10
            self.ood_samples = torch.from_numpy(np.load('/home1/gaoheng/gh_workspace/GOLDEN_HOOP/cifar10_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy')).cuda()   # (10, 1000, 768)
            self.ood_samples = self.ood_samples.reshape(10*1000, self.ood_samples.shape[2])
            self.N = 10*1000
            self.mi_loss = CLUB(768, 512, 512).cuda()
            self.n_data = 128
            
        elif self.dataset == 'cifar100':
            self.num_classes = 100
            if self.ood_fea_distill:
                self.ood_samples = torch.from_numpy(np.load('/home1/gaoheng/gh_workspace/GOLDEN_HOOP/cifar100_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy')).cuda()  # (10000, 768)
                self.ood_samples = self.ood_samples.reshape(100 * 10000, self.ood_samples.shape[2])
                self.N = 100*10000
                self.mi_loss = CLUB(768, 512, 512).cuda()
                self.n_data = 128
                # self.cl = CRDLoss(768, 512, self.n_data).cuda()
                
                # ipdb.set_trace()
                # print(self.ood_samples.shape)  # (1000000, 768)
        else:
            self.num_classes = 1000
            if self.ood_fea_distill:
                self.ood_samples = torch.from_numpy(np.load('/home1/gaoheng/gh_workspace/GOLDEN_HOOP/in1k_outlier_npos_embed_noise_0.07_select_50_KNN_300.npy')).cuda()  # (1000, 2500, 768)
                self.ood_samples = self.ood_samples.reshape(1000 * 2500, self.ood_samples.shape[2])
                self.N = 1000*2500
                self.mi_loss = CLUB(768, 512, 512).cuda()
                self.n_data = 128
            
        if fea_dist:
            self.dt = DT(in_dim=768, out_dim=512).cuda()
        
        # TO DO
        # Add ood feature distillation
        
 
        # load pretrained vit trained on cifar10  edadaltocg/vit_base_patch16_224_in21k_ft_cifar10
        # self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
        # timm.models._builder.build_model_with_cfg()
        # self.model = timm.create_model("timm/vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True, num_classes=10)
        # # self.model.head = nn.Linear(self.model.head.in_features, 10)
        # self.model.load_state_dict(
        #     torch.load('/home1/hezhuolin/9_gh_workspace/OpenOOD-main/OpenOOD-main/results/vit_cifar10_finetuned.bin')
        # )
        # self.model = ViTModel.from_pretrained('')
        if self.dataset == 'cifar10':
            self.model = timm.create_model("timm/vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=False)
            self.model.head = nn.Linear(self.model.head.in_features, self.num_classes)
            self.model.load_state_dict(
                # torch.hub.load_state_dict_from_url(
                #     "https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar10/resolve/main/pytorch_model.bin",
                #     map_location="cpu",
                #     file_name="vit_base_patch16_224_in21k_ft_cifar10.pth",
                # )
                torch.load('/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/pytorch_model.bin')
            )
        else:
            self.model = VisionTransformer(num_classes=self.num_classes)
            self.model.load_state_dict(
                torch.load('/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_vit_pretrained_finetune_trainer_e50_lr0.0001_default_cifar100_finetune_final_2/s0/best.ckpt')
            )
            
        
        self.kl_loss = nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=False)
        # if self.fea_distill:
        #     self.optimizer = torch.optim.SGD(
        #         [
        #             {
        #             'params': net.parameters()
        #             },
        #             {
        #             'params': self.dt.parameters(), 'lr':0.1
        #             }
        #         ],
        #         config.optimizer.lr,
        #         momentum=config.optimizer.momentum,
        #         weight_decay=config.optimizer.weight_decay,
        #         nesterov=True,
        #     )
        #     # self.optimizer = torch.optim.Adam(
        #     #     [
        #     #         {
        #     #         'params': net.parameters()
        #     #         },
        #     #         {
        #     #         'params': self.dt.parameters()
        #     #         }]
        #     #     ,
        #     #     config.optimizer.lr,
        #     #     # momentum=config.optimizer.momentum,
        #     #     # weight_decay=config.optimizer.weight_decay,
        #     #     # nesterov=True,
        #     # )
        # elif self.ood_fea_distill:
        #     # self.optimizer = torch.optim.SGD(
        #     #     [
        #     #         {
        #     #             'params': net.parameters()
        #     #         },
        #     #         {
        #     #             'params': self.dt.parameters(), 'lr': 0.1
        #     #         },
        #     #         {
        #     #             'params': self.cl.embed_t.parameters(), 'lr': 0.1
        #     #         }
        #     #     ],
        #     #     config.optimizer.lr,
        #     #     momentum=config.optimizer.momentum,
        #     #     weight_decay=config.optimizer.weight_decay,
        #     #     nesterov=True,
        #     # )
        self.optimizer = torch.optim.SGD(
            [
                {
                    'params': net.parameters()
                },
                {
                    'params': self.dt.parameters(), 'lr': 0.1
                },
                {
                    'params': self.mi_loss.p_mu.parameters(), 'lr': 0.1
                },
                {
                    'params': self.mi_loss.p_logvar.parameters(), 'lr': 0.1
                }
            ],
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )
        # else:
        #     self.optimizer = torch.optim.SGD(
        #         net.parameters(),
        #         config.optimizer.lr,
        #         momentum=config.optimizer.momentum,
        #         weight_decay=config.optimizer.weight_decay,
        #         nesterov=True,
        #     )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )
        # self.mid_fea_kd_loss = nn.MSELoss(reduce='mean')
        self.mid_fea_kd_loss = nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=False)

    # def forward_hook(self, module, inputs, outputs):
    #     feature_map_inputs.append(inputs)
    #     feature_map_outputs.append(outputs)

    def train_epoch(self, epoch_idx):
        if self.fea_distill:
            self.dt.train()
        self.net.train()
        self.model.cuda()
        self.model.eval()
        # self.model.head_drop.register_forward_hook(self.forward_hook)
        
        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)
        # teacher_train_dataiter = iter(self.train_loader2)
        
        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            
            if train_step == len(train_dataiter):
                continue
            # teacher_batch = next(teacher_train_dataiter)
            
            # teacher_data = teacher_batch['data'].cuda()
            # # print(teacher_data.shape)
            # teacher_target = teacher_batch['label'].cuda()
            # print(len(batch['data']))
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            # forward
            logits_classifier, feature = self.net(data, return_feature=True)
            log_soft = F.log_softmax(logits_classifier, dim=1)
            # print(log_soft.shape)
            # print(log_soft[0, :])
            # print(logits_classifier)
            # print(logits_classifier.shape)
            # softmax = nn.Softmax(dim=1)(logits_classifier)
            with torch.no_grad():
                vit_cls, mid_fea = self.model(data, return_feature=True)
                vit_cls = F.softmax(vit_cls, dim=1)
                # print(vit_cls.shape)
                # print(self.model.head_drop)
                # mid_fea = self.model.forward_features(data)
                # print(mid_fea.shape)
                
            # print(vit_cls[0, :])
            # print(vit_cls.shape)
            
            loss = F.cross_entropy(logits_classifier, target)
            
            if self.logits_distill:
                loss_kl = self.kl_loss(log_soft, vit_cls)
                loss += 4*loss_kl
                
            if self.fea_distill:
                # print(feature_map_outputs[0].shape)
                # kd_feature = self.dt(feature_map_outputs[0])
                kd_fea = self.dt(mid_fea)
                fea_log_soft = F.log_softmax(feature, dim=1)
                kd_fea = F.softmax(kd_fea, dim=1)
                # feature_map_outputs = []
                # print(kd_feature.shape)
                # mid_fea_kd = self.mid_fea_kd_loss(feature, kd_fea)
                mid_fea_kd = self.mid_fea_kd_loss(fea_log_soft, kd_fea)
                # print(mid_fea_kd)
                loss += 8*mid_fea_kd
            
            if self.ood_fea_distill:
                idx = self.ood_sub_samples = torch.randperm(self.N)[:self.n_data]
                selected_ood_samples = self.ood_samples[idx]  # random selected samples
                cdl = self.mi_loss.forward(selected_ood_samples, feature)
                # cdl = self.cl(mid_fea[0], selected_ood_samples)
                # print(cdl)
                loss += 0.1*cdl
                # self.embed
            
            # if self.ood_fea_distill:
                # step1 : sampling ood embeddings
                # sum_temp=0
                # for index in range(self.num_classes):
                #     sum_temp += 500
                # if sum_temp == self.num_classes * 500:
                #     # generate ood features
                #     for index in range(self.num_classes):
                #         ID = F.normalize(self.data_dict[index], p=2, dim=1)
                        
                #         print(index)
                #         # KNN-based generation
                #         for index1 in tqdm(range(100)):
                #             new_dis = MultivariateNormal(torch.zeros(768), torch.eye(768))
                #             negative_samples = new_dis.rsample((1500,))
                #             sample_point1, boundary_point = generate_outliers(ID,
                #                                                             input_index=KNN_index,
                #                                                             negative_samples=negative_samples,
                #                                                             ID_points_num=2,
                #                                                             K=300,
                #                                                             select=50,
                #                                                             cov_mat=0.07,
                #                                                             sampling_ratio=1.0,
                #                                                             pic_nums=50,
                #                                                             depth=768,
                #                                                             shift=0)
                #             if index1 == 0:
                #                 sample_point = sample_point1
                #             else:
                #                 sample_point = torch.cat([sample_point, sample_point1], 0)
                #             # ipdb.set_trace()
                #         if index == 0:
                #             ood_samples = [sample_point * self.anchor[index].norm()]
                #         else:
                #             ood_samples.append(sample_point * self.anchor[index].norm())
                #             # gh debug
                #             print("gh debug ood samples of one-shot:{}".format(len(ood_samples)))
                    
                #         print("gh debug, the shape of ood sample:{}".format(ood_samples[0].shape))
                        
                #     print("gh debug, total length of ood samples:{}".format(len(ood_samples)))
                        # ipdb.set_trace()
                
                
           

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            
            self.optimizer.step()
            self.scheduler.step()
            
            torch.cuda.empty_cache()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced
