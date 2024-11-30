# imports
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward


# WaveletLoss (impl. here)
class WaveletLoss(nn.Module):
    def __init__(self, wavelet="haar", level=1, loss_fn=nn.MSELoss(), w0=1.0, w1=1.0):
        super(WaveletLoss, self).__init__()
        self.dwt = DWTForward(J=level, wave=wavelet)  # Wavelet Transform
        self.loss_fn = loss_fn
        self.loss_coeffs = (w0, w1)

    def forward(self, pred, target):
        if self.loss_coeffs == (0, 0):
            return torch.zeros((), dtype=pred.dtype, requires_grad=True, device=pred.device)

        # Perform Wavelet Transform on predicted and target images
        pred_coeffs = self.dwt(pred)
        target_coeffs = self.dwt(target)

        # Compute loss for each subband
        total_loss = 0
        pred_yl, pred_yh = pred_coeffs
        target_yl, target_yh = target_coeffs

        for pred_coeffs, target_coeffs in zip(pred_yl, target_yl):
            total_loss += self.loss_fn(pred_coeffs, target_coeffs) * self.loss_coeffs[0]

        for pred_coeffs, target_coeffs in zip(pred_yh, target_yh):
            total_loss += self.loss_fn(pred_coeffs, target_coeffs) * self.loss_coeffs[1]

        return total_loss
