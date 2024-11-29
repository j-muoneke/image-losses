# imports
import torch.nn as nn

class ResNetSubset(nn.Module):
    """Store a subset of ResNet model weights to reduce memory usage."""
    def __init__(self, resnet):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1

        for param in self.parameters():
            param.requires_grad_(False)
    
    def forward(self, x):
        early_features = self.conv1(x)
        x = self.bn1(early_features)
        x = self.relu(x)
        x = self.maxpool(x)

        mid_features = self.layer1(x)
        return early_features, mid_features


# CNN Loss class
class CNNLoss(nn.Module):
    pass