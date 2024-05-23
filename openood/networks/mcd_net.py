import torch.nn as nn


class MCDNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super(MCDNet, self).__init__()

        self.backbone = backbone
        try:
            feature_size = backbone.feature_size
        except AttributeError:
            feature_size = backbone.module.feature_size

        self.fc1 = nn.Linear(feature_size, num_classes)
        self.fc2 = nn.Linear(feature_size, num_classes)

    # test conf
    def forward(self, x, return_double=False, return_feature=True):

        _, feature = self.backbone(x, return_feature=True)

        logits1 = self.fc1(feature)
        logits2 = self.fc2(feature)

        if return_double:
            if return_feature:
                return logits1, logits2, feature
            else:
                return logits1, logits2
        else:
            if return_feature:
                return logits1, feature
            return logits1
        
        # if return_feature:
        #     return logits1, feature
        # else:
        #     return logits1
