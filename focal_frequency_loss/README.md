# FFL Package Desc.

Contains a slight modification of the FFL PyTorch impelementation. Further description here + link to EndlessSora's original repo.

## Contents

- `focal_frequency_loss.py`: The main implementation of the FFL class.

## Installation

To use the CNN Loss class, ensure you have the following dependencies installed:

- Python 3.6+
- PyTorch (preferably) 1.7.0+

You can install the package using pip:

```bash
pip install git+https://github.com/j-muoneke/image-losses.git@main#subdirectory=focal_frequency_loss
```

## Usage

Below's a simple use case - producing a freq-domain distance/loss metric for two tensors:
```python

import torch
from focal_frequency_loss import FocalFrequencyLoss as FFL
device = "cuda:0" if torch.cuda_is_available() else "cpu"
# Initialize the FFL Loss
# Define an FFL metric, initialised for multipatching ([1, 2]), exponential down-weighting (a=0.95) and magnified (absolute) loss weighting
# Also tell it to run on the hardware environment's GPU (if available) with .to(device)
loss_fn = FFL(loss_weight = 10.0, alpha = 0.95, patch_factors = [1,2]).to(device)


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

Complete credit for implementation goes to the work present in https://github.com/EndlessSora/focal-frequency-loss/blob/master/focal_frequency_loss/focal_frequency_loss.py.
Linked repository also contains licensing, a link to the original paper (see here - https://arxiv.org/abs/2012.12821) and contribution credits.
