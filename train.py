"""Train the beta-VAE.

Usage:
    python train.py                     # use defaults from config.py
    python train.py --beta 8 --epochs 30
    python train.py --quick             # 1-epoch smoke test

After training, the following are saved:
    outputs/beta_vae.pt   model weights (with beta / latent_dim metadata)
    outputs/history.json  per-epoch loss / recon / kl curves
"""
import argparse
import json
import random
import time

import numpy as np
import torch

import config
from dataset import get_dataloaders
from model import BetaVAE


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref="auto"):
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--beta", type=float, default=None, help="beta coefficient (defaults to config.BETA)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--quick", action="store_true", help="1-epoch smoke test")
    return p.parse_args()


def main():
    args = parse_args()
    beta = args.beta if args.beta is not None else config.BETA
    epochs = 1 if args.quick else (args.epochs or config.EPOCHS)
    latent_dim = args.latent_dim or config.LATENT_DIM
    lr = args.lr if args.lr is not None else config.LR
    device = get_device(args.device or config.DEVICE)

    set_seed(config.SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[train] device={device} beta={beta} latent_dim={latent_dim} "
          f"epochs={epochs} lr={lr}")

    train_loader, _ = get_dataloaders(config)
    model = BetaVAE(latent_dim, config.HIDDEN_DIMS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"epoch": [], "loss": [], "recon": [], "kl": []}
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = ep_recon = ep_kl = 0.0
        t0 = time.time()
        for bi, (x, *_factors) in enumerate(train_loader):
            x = x.to(device)
            x_recon, mu, logvar = model(x)
            loss, recon, kl = model.loss(x, x_recon, mu, logvar, beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_loss += loss.item()
            ep_recon += recon.item()
            ep_kl += kl.item()
            if (bi + 1) % config.LOG_INTERVAL == 0:
                print(f"  epoch {epoch:>3}/{epochs} batch {bi + 1:>4} "
                      f"loss {loss.item():.3f} recon {recon.item():.3f} kl {kl.item():.3f}")

        n = len(train_loader)
        ep_loss, ep_recon, ep_kl = ep_loss / n, ep_recon / n, ep_kl / n
        history["epoch"].append(epoch)
        history["loss"].append(ep_loss)
        history["recon"].append(ep_recon)
        history["kl"].append(ep_kl)
        print(f"[epoch {epoch:>3}/{epochs}] loss {ep_loss:.4f} recon {ep_recon:.4f} "
              f"kl {ep_kl:.4f} ({time.time() - t0:.1f}s)")

    torch.save({"model_state": model.state_dict(),
                "beta": beta, "latent_dim": latent_dim}, config.MODEL_PATH)
    with open(config.HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[train] saved model -> {config.MODEL_PATH}")
    print(f"[train] saved history -> {config.HISTORY_PATH}")


if __name__ == "__main__":
    main()
