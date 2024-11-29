import torch as th
from torchvision.models import resnet101, resnet18, resnet34, resnet50, resnet152

import pytest

from . import CNNLoss


def test_cnn_loss_default_init() -> None:
    cnn_loss = CNNLoss()


def test_cnn_loss_init_from_string() -> None:
    cnn_loss = CNNLoss(model="ResNet34")


def test_cnn_loss_init_from_weights() -> None:
    cnn_loss = CNNLoss(model=resnet101())

MODELS = [("resnet18", resnet18), ("resnet34", resnet34), ("resnet50", resnet50), ("resnet101", resnet101), ("resnet152", resnet152)]

@pytest.mark.parametrize("model_name,model_type", MODELS)
def test_cnn_loss_different_models(model_name: str, model_type) -> None:
    cnn_loss = CNNLoss(model=model_name)
    model = model_type()
    assert cnn_loss._resnet.conv1.weight.data.shape == model.conv1.weight.data.shape
    assert not th.all(th.eq(cnn_loss._resnet.conv1.weight.data, model.conv1.weight.data))

@pytest.mark.parametrize("model_type", [model_type for _, model_type in MODELS])
def test_cnn_loss_different_models(model_type) -> None:
    model = model_type()
    cnn_loss = CNNLoss(model=model)
    assert th.all(th.eq(cnn_loss._resnet.conv1.weight.data, model.conv1.weight.data))

def test_cnn_loss_init_from_invalid_string() -> None:
    with pytest.raises(ValueError):
        cnn_loss = CNNLoss(model="unknown")

def test_cnn_loss_forward() -> None:
    real, recon = th.randn(2, 4, 3, 224, 224, dtype=th.float32)
    loss_fn = CNNLoss()
    loss = loss_fn(real, recon)
    assert loss.ndim == 0
    assert loss.dtype == th.float32
    assert th.isfinite(loss)

def test_cnn_loss_forward_zero_weight() -> None:
    real, recon = th.randn(2, 4, 3, 224, 224, dtype=th.float32)
    loss_fn = CNNLoss(w0=0.0, w1=0.0)
    loss = loss_fn(real, recon)
    assert loss.ndim == 0
    assert loss.dtype == th.float32
    assert th.allclose(loss, th.zeros_like(loss))

def test_cnn_loss_forward_equal_dists() -> None:
    real = th.randn(4, 3, 224, 224, dtype=th.float32)
    recon = real.clone()
    loss_fn = CNNLoss()
    loss = loss_fn(real, recon)
    assert loss.ndim == 0
    assert loss.dtype == th.float32
    assert th.allclose(loss, th.zeros_like(loss))

def test_cnn_loss_forward_weight_ratios() -> None:
    real, recon = th.randn(2, 4, 3, 224, 224, dtype=th.float32)
    loss_fn_1x = CNNLoss(w0=0.1, w1=1.0)
    loss_fn_10x = CNNLoss(w0=1.0, w1=10.0)
    loss_1x = loss_fn_1x(real, recon)
    loss_10x = loss_fn_10x(real, recon)
    assert th.allclose(loss_1x * 10.0, loss_10x)

def test_cnn_loss_forward_early_weight_monotonicity() -> None:
    real, recon = th.randn(2, 4, 3, 224, 224, dtype=th.float32)
    loss_fn_1x = CNNLoss(w0=1.0, w1=1.0)
    loss_fn_2x = CNNLoss(w0=2.0, w1=1.0)
    loss_1x = loss_fn_1x(real, recon)
    loss_2x = loss_fn_2x(real, recon)
    assert th.gt(loss_2x, loss_1x)
    assert th.lt(loss_2x, 2 * loss_1x)

def test_cnn_loss_forward_mid_weight_monotonicity() -> None:
    real, recon = th.randn(2, 4, 3, 224, 224, dtype=th.float32)
    loss_fn_1x = CNNLoss(w0=1.0, w1=1.0)
    loss_fn_2x = CNNLoss(w0=1.0, w1=2.0)
    loss_1x = loss_fn_1x(real, recon)
    loss_2x = loss_fn_2x(real, recon)
    assert th.gt(loss_2x, loss_1x)
    assert th.lt(loss_2x, 2 * loss_1x)
