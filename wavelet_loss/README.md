# DWTLoss Package Desc.

Contains a simple PyTorch wrapper for a Wavelet Loss implementation. Further Description Here.

## Contents

- `wavelet_loss.py`: The main implementation of the WaveletLoss class.

## Installation

To use the WaveletLoss class, ensure you have the following pip dependencies installed:

- python 3.6+
- torch (preferably 1.7.0+), with compatible torchvision deps
- pytorch-wavelets (python module = pywt)

You can install the package using pip:

```bash
pip install git+https://github.com/j-muoneke/image-losses.git@main#subdirectory=wavelet_loss
```

## Usage

Below's a simple use case - producing a freq-based Discrete Wavelet Transform Loss (DWTL) for two tensors:
```python

import torch
from wavelet_loss import WaveletLoss as DWTL
device = "cuda" if torch.cuda_is_available() else "cpu"

# Initialize the Wavelet Loss


# Define a WaveletLoss metric, 
# initialised to operate over the "haar" frequency bands,   (default : wavelet = "haar")
# with single-level decomposition,                          (default : level = 1)
# default loss coefficients,                                (default : w0 = w1 = 1)
# and Huber as the underlying loss function                 (default : loss_fn = nn.MSELoss())
loss_fn = DWTL(loss_fn = nn.HuberLoss()).to(device)


# Also tell it to run on the hardware environment's GPU (if available) with .to(device)
# Example tensors representing images
image1 = torch.randn(1, 3, 256, 256).to(device)  # Batch size of 1, 3 color channels, 256x256 image
image2 = torch.randn(1, 3, 256, 256).to(device)

# Calculate the loss
loss = loss_fn(image1, image2)
print(f'FFL Loss: {loss.item()}')
```

## License

This project is licensed under the MIT License. [TODO] See the `LICENSE` file for more details. [/TODO]

## Acknowledgements

Heavy reliance on Python-Wavelets for implementations of various wavelet funcs (Haar)
