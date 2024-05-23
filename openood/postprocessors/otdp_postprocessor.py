from typing import Any
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict
from openood.preprocessors.transform import normalization_dict


import numpy as np

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

def gaussian_kernel(x, sigma=0.1):
    # Compute the squared W1 distances
    squared_distances = np.sum(x ** 2, axis=0)
    
    # Apply the Gaussian kernel function
    K = np.exp(-0.5 * squared_distances / (sigma ** 2))
    
    return K

# # Example usage
# # Assuming we have two datasets, each containing 3 samples with 2 features each
# X = np.array([[1, 2], [3, 4], [5, 6]])
# X_prime = np.array([[7, 8], [9, 10], [11, 12]])
# sigma = 2.0  # You can adjust the value of sigma to control the width of the kernel

# # Calculate the Gaussian kernel matrix
# K = gaussian_kernel_matrix(X, X_prime, sigma)
# print("Gaussian Kernel Matrix:")
# print(K)

class OTDP(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = True
        self.T = 1000
        # try:
        #     self.input_std = normalization_dict[self.config.dataset.name][1]
        # except KeyError:
        #     self.input_std = [0.5, 0.5, 0.5]

    def w1_distance(self, p, q):
        return wasserstein_distance(p, q)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()

            print('Extracting id validation softmax posterior distributions')
            all_softmax = []
            preds = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['val'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    logits = net(data)
                    all_softmax.append(F.softmax(logits, 1).cpu())
                    preds.append(logits.argmax(1).cpu())

            all_softmax = torch.cat(all_softmax)
            preds = torch.cat(preds)

            self.mean_softmax_val = []
            for i in tqdm(range(self.num_classes)):
                # if there are no validation samples
                # for this category
                if torch.sum(preds.eq(i).float()) == 0:
                    temp = np.zeros((self.num_classes, ))
                    temp[i] = 1
                    self.mean_softmax_val.append(temp)
                else:
                    self.mean_softmax_val.append(
                        all_softmax[preds.eq(i)].mean(0).numpy())

            # self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # data.requires_grad = True
        logits, feat = net(data, return_feature=True)
        preds = logits.argmax(1)
        softmax = F.softmax(logits, 1).cpu().numpy()
        # print(feat.shape)
        # criterion = nn.CrossEntropyLoss()

        # labels = logits.detach().argmax(axis=1)

        # # Using temperature scaling
        # logits = logits / 1000

        # loss = criterion(logits, labels)
        # loss.requires_grad_(True)
        # loss.backward()

        # # Normalizing the gradient to binary in {0, 1}
        # gradient = torch.ge(data.grad.detach(), 0)
        # gradient = (gradient.float() - 0.5) * 2

        # # Scaling values taken from original code
        # gradient[:, 0] = (gradient[:, 0]) / self.input_std[0]
        # gradient[:, 1] = (gradient[:, 1]) / self.input_std[1]
        # gradient[:, 2] = (gradient[:, 2]) / self.input_std[2]

        # # Adding small perturbations to images
        # tempInputs = torch.add(data.detach(), gradient, alpha=-0.0014)
        # # # add noise
        # # noise = np.random.normal(0, 1, feat.shape)
        # # for _ in range(50):
        # #     noise_feat = feat.cpu().numpy() + noise 
        
        # # non-input ood maker
        # # ood_input = torch.zeros(data.shape).cuda()
        # # ood_logits, ood_feat = net(ood_input, return_feature=True)
        # # ood_feat = normalizer(ood_feat.cpu().numpy())
        # nlogits, feat = net(tempInputs, return_feature=True)


        ####### mv_sum = torch.zeros_like(feat).cuda()
        ######## # rank-k feature removal
        ######## u, s, v = torch.linalg.svd(feat, full_matrices=False)
        ####### for k in range(math.ceil(0.999*feat.shape[0])):
        #######     mv_sum += s[k]*torch.dot(u[k, :], v[:, k])

        # print("Debug, s shape:{}".format(s.shape))
        # s.shape [200] desc
        # print("debug, the shape of u:{}".format(u.shape)) # (200, 200)
        # print("debug, the shape of v:{}".format(v.shape))  # (200, 512)
        ######ood_feat = feat - mv_sum
        # ood_feat = normalizer(ood_feat.cpu().numpy())
        # print(ood_feat.shape)
        scores = np.zeros(feat.shape[0])
        feat = normalizer(feat.cpu().numpy())
        # ood_feat = normalizer((1/feat[0].shape[0])*np.ones_like(feat))
        # feat = feat.cpu().numpy()
        ######ood_feat = ood_feat.cpu().numpy()
        # noise_feat = normalizer(noise_feat)
        # softmax = softmax.cpu().numpy()
        # print(feat[0].shape)
        # print
        # N = feat.shape[0]
        for i in range(feat.shape[0]):
            # scores[i] = -np.exp(1000*self.w1_distance(feat[i], np.zeros_like(feat[i])))
            # scores[i] = gaussian_kernel(self.w1_distance(feat[i], np.zeros_like(feat[i])))
            # scores[i] = gaussian_kernel(100*self.w1_distance(feat[i], np.zeros_like(feat[i])))  # The best now
            
            # zero ood
            # scores[i] = gaussian_kernel(100*self.w1_distance(feat[i], np.zeros_like(feat[i])))
            # non-input ood
            # scores[i] = gaussian_kernel(self.w1_distance(feat[i], ood_feat[i]))
            # scores[i] = gaussian_kernel(100*(self.w1_distance(feat[i], np.zeros_like(feat[i]))))
            scores[i] = -10000*self.w1_distance(feat[i],
                                          # ood_feat[i]
                                         # np.zeros_like(feat[i])
                                          # -feat[i]
                                          # ood_feat[i]
                                          (1/(feat[i].shape[0]))*np.ones_like(feat[i])
                                          )
            # scores[i] = -self.w1_distance(feat[i], np.zeros_like(feat[i]))
            # scores[i] = gaussian_kernel(100*(self.w1_distance(feat[i], noise_feat[i])))
            # scores[i] = np.linalg.norm(feat[i]-np.zeros_like(feat[i]), ord=2)
            # scores[i] = gaussian_kernel(100*self.w1_distance(softmax[i], np.zeros_like(softmax[i])))
        
        # scores = -pairwise_distances_argmin_min(
        #     softmax, np.array(self.mean_softmax_val), metric=self.kl)[1]
        return preds, torch.from_numpy(scores)
