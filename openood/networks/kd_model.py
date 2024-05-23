import torch.nn as nn


class DT(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DT, self).__init__()
        '''
            A module for transfering ViT's penultimate layer's feature to ResNet-18's feature
        '''
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pre_norm = nn.BatchNorm1d(self.in_dim)
        self.mlp1 = nn.Linear(in_features=self.in_dim, out_features=self.out_dim, bias=True)
        self.act1 = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(self.out_dim)
        # self.mlp2 = nn.Linear(in_features=self.in_dim, out_features=self.in_dim, bias=True)
        # self.act2 = nn.ReLU()
        # self.norm2 = nn.BatchNorm1d(self.in_dim)
        # self.mlp3 = nn.Linear(in_features=self.in_dim, out_features=self.out_dim, bias=True)
        # self.act3 = nn.ReLU()
        # self.norm3 = nn.BatchNorm1d(self.out_dim)
        
    def forward(self, x):
        x = self.pre_norm(x)
        x = self.mlp1(x)
        x = self.act1(x)
        x = self.norm1(x)
    
        # x = self.mlp2(x)
        # x = self.act2(x)
        # x = self.norm2(x)

        # x = self.mlp3(x)
        # x = self.act3(x)
        # x = self.norm3(x)
        
        return x
        