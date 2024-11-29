# CNN Loss Package

This package contains a PyTorch implementation of a Convolutional Neural Network (CNN) Loss class designed for image-based neural networks. 
The CNN Loss is a learning perceptual loss metric. It achieves this with layers from a trained ResNet model.

## Contents

- `cnn_loss.py`: The main implementation of the CNN Loss class. Contains the CNNLoss and ResNetSubset classes, but one only need import the former. The latter is exposed for experimentation with ResNet layers.

## Installation

To use the CNN Loss class, ensure you have the following dependencies installed:

- Python 3.6+
- PyTorch (preferably) 1.7.0+
- torchvision 0.8.0+

You can install the package using pip:

```bash
pip install git+https://github.com/j-muoneke/image-losses.git@main#subdirectory=cnn_loss
```

## Usage

Below's a simple use case - producing a perceptual similarity metric for two images:
(Need George for this bit)
```python

import torch
from cnn_loss import CNNLoss

# Initialize the CNN Loss
loss_fn = CNNLoss()

# Example tensors representing images
image1 = torch.randn(1, 3, 256, 256).to(device)  # Batch size of 1, 3 color channels, 256x256 image
image2 = torch.randn(1, 3, 256, 256).to(device)

# Calculate the loss
loss = loss_fn(image1, image2)
print(f'Perceptual Loss: {loss.item()}')
```

## License

This project is licensed under the MIT License. [TODO] See the `LICENSE` file for more details. [/TODO]

## Acknowledgements

This implementation is inspired by... (where'd we get this idea, again?)
