from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from .base_postprocessor import BasePostprocessor
# import ot
import torch.nn.functional as F

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


class WDPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(WDPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.K = self.args.K
        self.activation_log = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False
        self.epsilon = 0.000001

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            activation_log = []
            net.eval()
            w1_distances =[]
            fd = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()

                    _, feature = net(data, return_feature=True)
                    # print("The second dimension of feature:{}".format(feature.shape[1]))  512
                    activation_log.append(
                        normalizer(feature.data.cpu().numpy()))

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.feat_batch_mean = np.mean(self.activation_log, axis=0)
            self.feat_batch_mean = normalizer(self.feat_batch_mean)
            # for b in tqdm(range(self.activation_log.shape[0])):
            #     w1_distances.append(wasserstein_distance(self.feat_batch_mean, self.activation_log[b]))
            
            #     # fd.append(frechet_distance(self.feat_batch_mean, self.activation_log[b]))
            # w1_distances = np.array(w1_distances)
            # # fd = np.array(fd)
            # print("W1 Distances:{}".format(w1_distances))
            # # print("Frechet Distance:{}".format(fd))
            # max_k = np.argmax(w1_distances)
            # # max_k = np.argmax(fd)
            # # print("max_k:{}".format(max_k))
            # self.max_distance = w1_distances[max_k]  # max w1 distance
            # # self.max_distance = fd[max_k]
            # print("max distance:{}".format(self.max_distance))
            # self.max_vec = self.activation_log[max_k]
            # # print("max_vec:{}".format(self.max_vec))
            # self.threshold = self.max_distance + self.epsilon
            # print("Threshold:{}".format(self.threshold))
            # print(self.activation_log.shape)  # (50000, 512)
            # ipdb.set_trace()
            # self.index = faiss.IndexFlatL2(feature.shape[1])
            # self.index.add(self.activation_log)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, feature = net(data, return_feature=True)
        feature_normed = normalizer(feature.data.cpu().numpy())
        conf = torch.zeros(feature_normed.shape[0])
        for i in range(feature_normed.shape[0]):
            # conf[i] = torch.mean(torch.nn.functional.softmax(torch.from_numpy(feature_normed[i]), dim=0))-torch.mean(torch.nn.functional.softmax(torch.from_numpy(self.feat_batch_mean), dim=0)) # / self.threshold
            # conf[i] = wasserstein_distance(torch.nn.functional.softmax(torch.from_numpy(feature_normed[i]), dim=0).numpy(), torch.nn.functional.softmax(torch.from_numpy(self.feat_batch_mean), dim=0).numpy())
                    # 计算每个样本的Wasserstein距离
            # current_distance = wasserstein_distance(
            #     torch.nn.functional.log_softmax(torch.from_numpy(feature_normed[i]), dim=0).numpy(),
            #     torch.nn.functional.log_softmax(torch.from_numpy(self.feat_batch_mean), dim=0).numpy()
            # )
            conf[i] = wasserstein_distance(feature_normed[i], self.feat_batch_mean)
            # conf[i] = ot.emd2_id(
            #     torch.nn.functional.softmax(torch.from_numpy(feature_normed[i]), dim=0).numpy(),
            #     torch.nn.functional.softmax(torch.from_numpy(self.feat_batch_mean), dim=0).numpy()
            # )
            # conf[i] = frechet_distance(feature_normed[i], self.feat_batch_mean)
            
            # conf[i] = 1 / (1 + np.exp(-(current_distance - self.threshold)))
            # 利用阈值进行二元分类
            # conf[i] = -1 if current_distance >= self.threshold else 1
        # print(kth_dist.shape) # (200)  / self.threshold
        # conf = torch.from_numpy(conf)
        # print(conf.shape)
        # print("gh debug:{}".format(type(conf)))
        # _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        # D, _ = self.index.search(
        #     feature_normed,
        #     self.K,
        # )
        # kth_dist = -D[:, -1]
        # print(kth_dist)
        # print(kth_dist.shape) # (200)
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return pred, conf # torch.from_numpy(kth_dist)

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K
