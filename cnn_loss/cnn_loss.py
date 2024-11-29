# imports
from typing import ClassVar, List, Tuple, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as vmodels

class ResNetSubset(nn.Module):
    """Store a subset of ResNet model weights to reduce memory usage."""
    def __init__(self, resnet) -> None:
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1

        for param in self.parameters():
            param.requires_grad_(False)
    
    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        early_features = self.conv1(x)
        x = self.bn1(early_features)
        x = self.relu(x)
        x = self.maxpool(x)

        mid_features = self.layer1(x)
        return early_features, mid_features


# CNN Loss class
class CNNLoss(nn.Module):
    VALID_MODELS: ClassVar[List[str]] = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    def __init__(self, model: Union[str, nn.Module] = "Resnet18", w0=1.0, w1=0.03) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            model = model.lower()
            if model not in self.VALID_MODELS:
                raise ValueError(f"Invalid model name {model}, valid models are {self.VALID_MODELS}")
            model = getattr(vmodels, model)(weights="DEFAULT")
        self.loss_weights = (w0, w1)
        if self.loss_weights != (0.0, 0.0):
            self._resnet = ResNetSubset(model)
    
    def forward(self, recon: th.Tensor, real) -> th.Tensor:
        early_loss_w, mid_loss_w = self.loss_weights
        if early_loss_w == 0.0 and mid_loss_w == 0.0:
            return th.zeros((), dtype=recon.dtype, device=recon.device, requires_grad=True)

        recon_early_features, recon_mid_features = self._resnet(recon)
        real_early_features, real_mid_features = self._resnet(real)
        
        return F.mse_loss(recon_early_features, real_early_features) * early_loss_w + F.mse_loss(recon_mid_features, real_mid_features) * mid_loss_w