import torch
import torch.nn as nn

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft
    
class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements a patch-variable version of Focal Frequency Loss - a
    frequency domain loss function for optimizing generative models.

    Original Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>
    
    Modified Ref:
    Application of Frequency Rectification on Images
    <TBA>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factors (int): the factor list to crop image patches for patch-based focal frequency loss. Default: 1, inverse-weighted (weight_i = norm(1/patch_factors[i])
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factors=[1], ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factors = patch_factors
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        
        # Naive - define per-patch freq. loss weights via weighted sum of all patch factors 
        nf_tmp = sum(patch_factors)
        self.patch_weights = list(reversed([x/nf_tmp for x in patch_factors]))

    def tensor2freq(self, x):
        freqs = []
        for patch_factor in self.patch_factors:
            _, _, h, w = x.shape
            assert h % patch_factor == 0 and w % patch_factor == 0, (
                'Patch factor should be divisible by image height and width')
            patch_list = []
            patch_h = h // patch_factor
            patch_w = w // patch_factor
            for i in range(patch_factor):
                for j in range(patch_factor):
                    patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

            # stack to Torch tensor
            y = torch.stack(patch_list, 1)

            # perform 2D DFT (real-to-complex, orthonormalization)
            if IS_HIGH_VERSION:
                freq = torch.fft.fft2(y, norm='ortho')
                freq = torch.stack([freq.real, freq.imag], -1)
            else:
                freq = torch.rfft(y, 2, onesided=False, normalized=True)

            freqs.append(freq)
        
        # stack and return    
        return freqs

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freqs = self.tensor2freq(pred)
        target_freqs = self.tensor2freq(target)
        
        # sanity check, of course
        assert len(self.patch_weights) == len(pred_freqs), (
            "Number of patch factors and patch_weights must match")
        
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # loop over patch_factor dim (list in tensor2freq)
        for i in range(len(self.patch_factors)):
            
            # splices for both SHOULD have shape: (N, C, H, W)
            total_loss +=  self.loss_formulation(pred_freqs[i], target_freqs[i], matrix) * self.patch_weights[i]
            
        return (total_loss * self.loss_weight)