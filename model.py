"""β-VAE 模型定义。

结构与标准 VAE 完全相同，唯一的区别在于损失函数中 KL 散度项前的系数 β。
β > 1 时加强对隐空间的正则约束，迫使模型将不同语义因子分配到不同维度，
从而实现特征解耦（feature disentanglement）。
"""
import torch
import torch.nn as nn


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """重参数化技巧：z = mu + eps * exp(0.5 * logvar)，使采样可反向传播。"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class Encoder(nn.Module):
    """卷积编码器：输入 1×28×28，输出 (mu, logvar)，维度 = latent_dim。

    下采样路径：28 -> 14 -> 7 -> 4。
    """

    def __init__(self, latent_dim: int, hidden_dims=(32, 64, 128)):
        super().__init__()
        layers, in_ch = [], 1
        # (kernel, stride, padding) 依次对应 28->14, 14->7, 7->4
        specs = [(4, 2, 1), (4, 2, 1), (3, 2, 1)]
        for h, (k, s, p) in zip(hidden_dims, specs):
            layers += [nn.Conv2d(in_ch, h, k, s, p), nn.BatchNorm2d(h), nn.LeakyReLU(0.2)]
            in_ch = h
        self.conv = nn.Sequential(*layers)
        self.flat_dim = hidden_dims[-1] * 4 * 4
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """转置卷积解码器：latent_dim -> 1×28×28。

    上采样路径：4 -> 7 -> 14 -> 28。
    """

    def __init__(self, latent_dim: int, hidden_dims=(32, 64, 128)):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.flat_dim = hidden_dims[-1] * 4 * 4
        self.fc = nn.Linear(latent_dim, self.flat_dim)

        layers, in_ch = [], hidden_dims[-1]
        # 4->7, 7->14, 14->28（最后一层输出 1 通道，不含激活/BN）
        specs = [(3, 2, 1), (4, 2, 1), (4, 2, 1)]
        for i, (h, (k, s, p)) in enumerate(zip(reversed(hidden_dims[:-1]), specs[:-1])):
            layers += [nn.ConvTranspose2d(in_ch, h, k, s, p), nn.BatchNorm2d(h), nn.LeakyReLU(0.2)]
            in_ch = h
        k, s, p = specs[-1]
        layers += [nn.ConvTranspose2d(in_ch, 1, k, s, p)]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc(z).view(-1, self.hidden_dims[-1], 4, 4)
        return torch.sigmoid(self.deconv(h))


class BetaVAE(nn.Module):
    """β-VAE：encoder + decoder。

    `loss` 返回 (总损失, 重构损失, KL 散度)。
    """

    def __init__(self, latent_dim: int, hidden_dims=(32, 64, 128)):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, hidden_dims)
        self.decoder = Decoder(latent_dim, hidden_dims)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss(self, x, x_recon, mu, logvar, beta=4.0):
        # 逐样本 BCE 求和后取平均，等价于平均负对数似然（伯努利）
        recon = nn.functional.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)
        # 每个样本与标准正态先验的 KL 散度，再取平均
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return recon + beta * kl, recon, kl
