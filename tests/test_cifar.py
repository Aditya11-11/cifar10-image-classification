from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import load_arrays
from src.gradcam import GradCAM
from src.models import EXPERIMENTS, BaselineCNN, CifarNet, build_model, count_params
from src.predict import DEFAULT_WEIGHTS, to_cifar

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


@pytest.mark.parametrize("name", list(EXPERIMENTS))
def test_every_experiment_builds_and_outputs_10_logits(name):
    model = build_model(**EXPERIMENTS[name]).eval()
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_ablation_switches_really_switch():
    no_bn = CifarNet(use_bn=False, dropout=0.3)
    no_do = CifarNet(use_bn=True, dropout=0.0)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in no_bn.modules())
    assert not any(isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)) for m in no_do.modules())


def test_global_pooling_keeps_cifarnet_small():
    # Dense layers on flattened feature maps are where the parameters usually go
    assert count_params(CifarNet()) < 2 * count_params(BaselineCNN())


def test_gradcam_heatmap_shape_and_range():
    model = CifarNet().eval()
    heat, cls, probs = GradCAM(model, model.last_conv)(torch.randn(1, 3, 32, 32))
    assert heat.shape == (32, 32) and 0 <= heat.min() and heat.max() <= 1
    assert 0 <= cls < 10 and np.isclose(probs.sum(), 1, atol=1e-5)


def test_any_image_becomes_32x32():
    assert to_cifar(Image.new("RGB", (640, 480))).size == (32, 32)


def test_data_arrays():
    x, y = load_arrays("test")
    assert x.shape == (10_000, 32, 32, 3) and x.dtype == np.uint8
    assert np.bincount(y).tolist() == [1000] * 10


@pytest.mark.skipif(not DEFAULT_WEIGHTS.exists(), reason="train the final model first")
def test_sample_images():
    from src.predict import classify

    files = sorted(SAMPLES.glob("*.png"))
    correct = sum(classify(Image.open(f))[0] == f.stem.split("_")[0] for f in files)
    assert correct >= len(files) - 2
