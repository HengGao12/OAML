from typing import Any
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import openood.utils.comm as comm



class BasePostprocessor:
    def __init__(self, config):
        self.config = config
        # self.if_wd = wd

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, feat = net(data, return_feature=True)
        # output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        pred_list, conf_list, label_list = [], [], []  # here is the original code
        # pred_list, conf_list, label_list, feat_list = [], [], [], []
        
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            
            # print("The shape of data:{}".format(data.shape))
            # print("The shape of label:{}".format(label.shape))

            # pred, conf, feat = self.postprocess(net, data)
            pred, conf = self.postprocess(net, data)
            # print(len(pred))
            # ipdb.set_trace()
            # pred, conf = self.postprocess(net, data)
            
            # print("The shape of pred:{}".format(pred.shape))
            # print("The shape of conf:{}".format(conf.shape))
            
            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())
            
            # feat_list.append(feat.cpu())
            
            # pred_list.append(pred).cpu())
            # conf_list.append(torch.from_numpy(conf).cpu())
            # label_list.append(torch.from_numpy(label).cpu())

        # convert values into numpy array
        # pred_list = torch.cat(pred_list).numpy().astype(int)
        # conf_list = torch.cat(conf_list).numpy()
        # label_list = torch.cat(label_list).numpy().astype(int)
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)
        # feat_list = torch.cat(feat_list).numpy()
        
        
        # print("ood conf1:{}".format(conf_list))
        return pred_list, conf_list, label_list   # , feat_list
        # return pred_list, conf_list, label_list
