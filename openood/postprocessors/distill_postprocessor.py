from typing import Any
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance 
from .base_postprocessor import BasePostprocessor
import ipdb
# import ot

class DistillPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(DistillPostprocessor, self).__init__(config)
        self.config = config
        self.args = self.config.postprocessor.postprocessor_args
        self.batch_size = 200
    
    def gaussian_kernel(self, x, y):
        return torch.exp(-torch.norm(x - y, 2)**2 / (2 * 0.1**2))
    
    def wd_ot(self, samples_real, samples_generated):
        distance_matrix = ot.dist(samples_real, samples_generated)
        return ot.emd2([], [], distance_matrix)
    
    def wd(self, x, y):
        return wasserstein_distance(x, y) 
    
    def z_score_normalize(self, arr):
        mean = np.mean(arr)
        std = np.std(arr)
        normalized_arr = (arr - mean) / std
        return normalized_arr

    def min_max_normalize(self, arr, feature_range=(-1, 1)):
        min_val = np.min(arr)
        max_val = np.max(arr)
        normalized_arr = (arr - min_val) / (max_val - min_val)
        normalized_arr = normalized_arr * (feature_range[1] - feature_range[0]) + feature_range[0]
        return normalized_arr
    
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # confs = []
        # print(net)
        # ipdb.set_trace()
        # weight = net.layer4[1].conv2.weight.data
        # B, C, H, W = weight.size()
        # weight = weight.squeeze()
        # weight_sub = power_iteration(weight.unsqueeze(0), iter=100)
        # weight = weight - weight_sub.squeeze()
        # weight = weight.view(B, C, H, W)
        # net.layer4[1].conv2.weight.data = weight

        # weight = net.module.body.block4[-1].conv2.weight.data
        # B, C, H, W = weight.size()
        # weight = weight.squeeze()
        # print(weight)
        # weight_sub = power_iteration(weight.unsqueeze(0), iter=100)
        # weight = weight - weight_sub.squeeze()
        # weight = weight.view(B, C, H, W)
        # net.layer4[1].conv1.weight.data = weight
        
        inputs = data.cuda()

        feat = net.intermediate_forward(inputs, layer_index=4)
        ori_feat = feat
        # print(feat.shape)
        B, C, H, W = feat.size()
        wd_score = np.zeros(B)
        conf = np.zeros(B)
        feat = feat.view(B, C, H * W)
        origin_feat = feat
        x1 = origin_feat.reshape(B, C*H*W)
        u,s,v = torch.linalg.svd(feat,full_matrices=False)
        # print(s.shape)
        # ipdb.set_trace()
        # feat = feat - s[:,0:1].unsqueeze(2)*u[:,:,0:1].bmm(v[:,0:1,:])
        #if you want to use PI for acceleration, comment the above 2 lines and uncomment the line below
        feat = feat - power_iteration(feat, iter=20)
        mfeat = feat
        x2 = mfeat.reshape(B, C*H*W)
        feat = feat.view(B,C,H,W)
        # logits = net.module.head(net.module.before_head(feat))
        logits = net.fc(torch.flatten(net.avgpool(ori_feat), 1))
        
        # conf = temperature * torch.logsumexp(logits / temperature, dim=1)
        # conf = 1 - 0.5*(1+torch.nn.functional.cosine_similarity(x1, x2, dim=1, eps=1e-8))
        # conf = 1 - self.gaussian_kernel(x1, x2)
        # x1 = torch.mean(x1, dim=0)
        # print(x1.shape)
        # x2 = torch.mean(x2, dim=0)
        for b in range(x1.shape[0]):
            wd_score[b] = self.wd(x1[b].cpu().numpy(), x2[b].cpu().numpy())
        
  

        wd_score = self.min_max_normalize(wd_score)
        # print("The wd_score:{}".format(wd_score))
        for b in range(x1.shape[0]):
            conf[b] = 0.5+0.5*wd_score[b]
        # print("The ood_conf is :{}".format(conf))
        conf = torch.from_numpy(conf)
        # conf = self.wd(x1.cpu().numpy(), x2.cpu().numpy())
        
        # print("Debug, the shape of conf:{}".format(conf.shape))
        
        _, pred = torch.max(logits, dim=1)
        # print("Debug, the shape of pred:{}".format(pred.shape))
        # confs.extend(conf.data.cpu().numpy())
        return pred, conf
    # def postprocess(self, net: nn.Module, data: Any):
    #     inputs = data.cuda()

    #     # Logit of Block 4 feature
    #     feat1 = net.intermediate_forward(inputs, layer_index=4)
    #     B, C, H, W = feat1.size()
    #     feat1 = feat1.view(B, C, H * W)
    #     origin_feat1 = feat1
    #     x1 = origin_feat1.reshape(B, -1)
    #     if self.args.accelerate:
    #         feat1 = feat1 - power_iteration(feat1, iter=20)
    #         x2 = feat1.reshape(B, -1)
    #     else:
    #         u, s, v = torch.linalg.svd(feat1, full_matrices=False)
    #         feat1 = feat1 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(
    #             v[:, 0:1, :])
    #         x2 = feat1.reshape(B, -1)
            
    #     origin_feat1 = origin_feat1.view(B, C, H, W)
    #     logits1 = net.fc(torch.flatten(net.avgpool(origin_feat1), 1))
    #     conf1 = 1 - 0.5*(1-torch.nn.functional.cosine_similarity(x1, x2, dim=1, eps=1e-8))

    #     # Logit of Block 3 feature
    #     feat2 = net.intermediate_forward(inputs, layer_index=3)
    #     B, C, H, W = feat2.size()
    #     feat2 = feat2.view(B, C, H * W)
    #     origin_feat2 = feat2
    #     x3 = origin_feat2.reshape(B, -1)
    #     if self.args.accelerate:
    #         feat2 = feat2 - power_iteration(feat2, iter=20)
    #         x4 = feat2.reshape(B, -1)
    #     else:
    #         u, s, v = torch.linalg.svd(feat2, full_matrices=False)
    #         feat2 = feat2 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(
    #             v[:, 0:1, :])
    #         x4 = feat2.reshape(B, -1)
        
    #     conf2 = 1 - 0.5*(1+torch.nn.functional.cosine_similarity(x3, x4, dim=1, eps=1e-8))

    #     origin_feat2 = origin_feat2.view(B, C, H, W)
    #     origin_feat2 = net.layer4(origin_feat2)
    #     logits2 = net.fc(torch.flatten(net.avgpool(origin_feat2), 1))

    #     # Fusion at the logit space
    #     logits = (logits1 + logits2) / 2
    #     # Fusion the confidence score
    #     conf = (conf1 + conf2) / 2

    #     _, pred = torch.max(logits, dim=1)
    #     return pred, conf



def _l2normalize(v, eps=1e-10):
    return v / (torch.norm(v, dim=2, keepdim=True) + eps)


# Power Iteration as SVD substitute for acceleration
def power_iteration(A, iter=20):
    u = torch.FloatTensor(1, A.size(1)).normal_(0, 1).view(
        1, 1, A.size(1)).repeat(A.size(0), 1, 1).to(A)
    v = torch.FloatTensor(A.size(2),
                          1).normal_(0, 1).view(1, A.size(2),
                                                1).repeat(A.size(0), 1,
                                                          1).to(A)
    for _ in range(iter):
        v = _l2normalize(u.bmm(A)).transpose(1, 2)
        u = _l2normalize(A.bmm(v).transpose(1, 2))
    sigma = u.bmm(A).bmm(v)
    # print(sigma)
    sub = sigma * u.transpose(1, 2).bmm(v.transpose(1, 2))
    return sub
