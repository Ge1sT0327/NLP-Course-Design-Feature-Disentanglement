"""FactorMNIST dataset.

Two interpretable "latent factors of variation" are applied on top of MNIST:
  1. rotation angle (continuous, in degrees)
  2. stroke thickness (discrete level 1..THICKNESS_LEVELS)

After training a beta-VAE on this data, we can quantitatively check whether
different latent dimensions have separately learned the "class / rotation /
thickness" factors, thereby verifying feature disentanglement.
"""
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms.functional as TF


def dilate(x: torch.Tensor, steps: int) -> torch.Tensor:
    """Approximate morphological dilation via max-pooling to simulate thicker strokes.

    Applies `steps` rounds of 3x3 max-pooling (stride=1, padding=1) to a 1xHxW
    binary image. Returns the input unchanged when steps=0.
    """
    for _ in range(steps):
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x


class FactorMNIST(Dataset):
    """MNIST with rotation / thickness factors.

    Each sample returns (image, class, rotation, thickness), all torch.Tensor:
      image     : [1, 28, 28] float, binarized + augmented image
      class     : [1] long, digit class 0..9
      rotation  : [1] float, the actual sampled rotation angle (degrees)
      thickness : [1] long, thickness level 1..THICKNESS_LEVELS
    """

    def __init__(self, root, train=True, rotation_range=(-30.0, 30.0),
                 thickness_levels=3, download=True):
        base = datasets.MNIST(root, train=train, download=download)
        self.imgs = base.data.float() / 255.0      # [N, 28, 28]
        self.labels = base.targets                 # [N]
        self.rotation_range = rotation_range
        self.thickness_levels = thickness_levels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x = self.imgs[idx]
        label = int(self.labels[idx])

        # 1) Binarize: make stroke contours sharp so rotation / thickness are more salient
        x = (x > 0.5).float().unsqueeze(0)         # [1, 28, 28]

        # 2) Stroke thickness factor
        thickness = random.randint(1, self.thickness_levels)
        x = dilate(x, steps=thickness - 1)

        # 3) Rotation factor
        angle = random.uniform(*self.rotation_range)
        x = TF.rotate(x, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
        x = x.clamp(0.0, 1.0)

        return (x,
                torch.tensor(label, dtype=torch.long),
                torch.tensor(angle, dtype=torch.float32),
                torch.tensor(thickness, dtype=torch.long))


def get_dataloaders(config):
    """Build the train / test DataLoaders."""
    common = dict(root=str(config.DATA_DIR), rotation_range=config.ROTATION_RANGE,
                  thickness_levels=config.THICKNESS_LEVELS)
    train_ds = FactorMNIST(train=True, **common)
    test_ds = FactorMNIST(train=False, **common)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS)
    return train_loader, test_loader


def load_clean_mnist(root, train=False, limit=None):
    """Load unaugmented (binarized only) MNIST images for reconstruction / traversal display."""
    base = datasets.MNIST(root, train=train, download=True)
    imgs = (base.data.float() / 255.0 > 0.5).float()   # [N, 28, 28]
    labels = base.targets
    if limit:
        imgs, labels = imgs[:limit], labels[:limit]
    return imgs.unsqueeze(1), labels                     # [N, 1, 28, 28]
