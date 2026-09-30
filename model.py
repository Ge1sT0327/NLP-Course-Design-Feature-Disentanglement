"""beta-VAE model definition.

The architecture is identical to a standard VAE; the only difference is the
coefficient beta in front of the KL term in the loss. When beta > 1, the
regularization on the latent space is strengthened, forcing the model to assign
different semantic factors to different dimensions and thereby achieving feature
disentanglement.
"""
import torch
import torch.nn as nn


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick: z = mu + eps * exp(0.5 * logvar), so that sampling is differentiable."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class Encoder(nn.Module):
    """Convolutional encoder: 1x28x28 input -> (mu, logvar) of size latent_dim.

    Downsampling path: 28 -> 14 -> 7 -> 4.
    """

    def __init__(self, latent_dim: int, hidden_dims=(32, 64, 128)):
        super().__init__()
        layers, in_ch = [], 1
        # (kernel, stride, padding) corresponds to 28->14, 14->7, 7->4
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
    """Transposed-convolutional decoder: latent_dim -> 1x28x28.

    Upsampling path: 4 -> 7 -> 14 -> 28.
    """

    def __init__(self, latent_dim: int, hidden_dims=(32, 64, 128)):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.flat_dim = hidden_dims[-1] * 4 * 4
        self.fc = nn.Linear(latent_dim, self.flat_dim)

        layers, in_ch = [], hidden_dims[-1]
        # 4->7, 7->14, 14->28 (the last layer outputs 1 channel, without activation/BN)
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
    """beta-VAE: encoder + decoder.

    `loss` returns (total loss, reconstruction loss, KL divergence).
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
        # Per-sample BCE summed and averaged, equivalent to the mean negative log-likelihood (Bernoulli)
        recon = nn.functional.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)
        # KL divergence between each sample and the standard normal prior, averaged
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return recon + beta * kl, recon, kl
