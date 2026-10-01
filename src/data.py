"""CIFAR-10 loaders: 50k training images (45k train / 5k validation) and 10k test images, 32x32 RGB.

The data comes from the Hugging Face mirror (uoft-cs/cifar10) because it downloads
~50x faster than the original Toronto server. The first run decodes the PNGs once
and caches them as a .npz file (~180 MB) so later runs start instantly.
"""
import io
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Per-channel mean/std of the CIFAR-10 training set
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

NORMALIZE = transforms.Normalize(MEAN, STD)
EVAL_TRANSFORM = transforms.Compose([transforms.ToTensor(), NORMALIZE])
# The two augmentations that matter most on CIFAR: shift the image around a bit
# (pad + random crop) and mirror it. A flipped cat is still a cat.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    NORMALIZE,
])


def load_arrays(split: str):
    """Return (images uint8 [N,32,32,3], labels int64 [N]) for 'train' or 'test'."""
    cache = DATA_DIR / f"cifar10_{split}.npz"
    if not cache.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        parquet = DATA_DIR / f"{split}.parquet"
        if not parquet.exists():
            print(f"Downloading CIFAR-10 {split} split ...")
            urlretrieve(HF_URL.format(split=split), parquet)
        df = pd.read_parquet(parquet)
        images = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in df["img"]])
        np.savez(cache, images=images, labels=df["label"].to_numpy(np.int64))
    data = np.load(cache)
    return data["images"], data["labels"]


class CIFAR10Arrays(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), int(self.labels[i])


def get_loaders(batch_size=128, augment=True, val_size=5000, seed=42, num_workers=4):
    train_x, train_y = load_arrays("train")
    test_x, test_y = load_arrays("test")
    train_full = CIFAR10Arrays(train_x, train_y, TRAIN_TRANSFORM if augment else EVAL_TRANSFORM)
    val_view = CIFAR10Arrays(train_x, train_y, EVAL_TRANSFORM)  # validation never sees augmentation
    test = CIFAR10Arrays(test_x, test_y, EVAL_TRANSFORM)

    perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(seed)).tolist()
    train_idx, val_idx = perm[val_size:], perm[:val_size]
    kw = dict(batch_size=batch_size, num_workers=num_workers, persistent_workers=num_workers > 0)
    return (DataLoader(Subset(train_full, train_idx), shuffle=True, drop_last=True, **kw),
            DataLoader(Subset(val_view, val_idx), **kw),
            DataLoader(test, **kw))


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo NORMALIZE for plotting. Accepts (3,H,W) or (N,3,H,W)."""
    mean = torch.tensor(MEAN).view(-1, 1, 1)
    std = torch.tensor(STD).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)
