"""FactorMNIST 数据集。

在 MNIST 基础上叠加两个可解释的“潜在变化因子”：
  1. 旋转角度（rotation，连续值，单位：度）
  2. 笔画粗细（thickness，离散等级 1..THICKNESS_LEVELS）

这样训练 β-VAE 后，就能定量检验不同隐变量维度是否分别学会了
“类别 / 旋转 / 粗细”这些独立因子，从而验证特征解耦。
"""
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms.functional as TF


def dilate(x: torch.Tensor, steps: int) -> torch.Tensor:
    """用 max-pooling 近似形态学膨胀，模拟“笔画变粗”。

    对 1×H×W 的二值图做 steps 次 3×3 max-pool（stride=1, padding=1）。
    steps=0 时原样返回。
    """
    for _ in range(steps):
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x


class FactorMNIST(Dataset):
    """带旋转 / 粗细因子的 MNIST。

    每个样本返回 (image, class, rotation, thickness)，均为 torch.Tensor：
      image     : [1, 28, 28] float，二值化 + 增强后的图像
      class     : [1] long，0..9 数字类别
      rotation  : [1] float，实际采样的旋转角度（度）
      thickness : [1] long，粗细等级 1..THICKNESS_LEVELS
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

        # 1) 二值化：让笔画轮廓清晰，旋转 / 粗细因子更显著
        x = (x > 0.5).float().unsqueeze(0)         # [1, 28, 28]

        # 2) 笔画粗细因子
        thickness = random.randint(1, self.thickness_levels)
        x = dilate(x, steps=thickness - 1)

        # 3) 旋转因子
        angle = random.uniform(*self.rotation_range)
        x = TF.rotate(x, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
        x = x.clamp(0.0, 1.0)

        return (x,
                torch.tensor(label, dtype=torch.long),
                torch.tensor(angle, dtype=torch.float32),
                torch.tensor(thickness, dtype=torch.long))


def get_dataloaders(config):
    """构建训练 / 测试 DataLoader。"""
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
    """加载未增强（仅二值化）的 MNIST 图像，用于重构 / 遍历展示。"""
    base = datasets.MNIST(root, train=train, download=True)
    imgs = (base.data.float() / 255.0 > 0.5).float()   # [N, 28, 28]
    labels = base.targets
    if limit:
        imgs, labels = imgs[:limit], labels[:limit]
    return imgs.unsqueeze(1), labels                     # [N, 1, 28, 28]
