"""Evaluation and visualization: quantitative metrics + report figures.

Outputs to outputs/:
  reconstructions.png    original vs. reconstructed images
  latent_traversal.png   per-dimension latent traversal (core figure, matching the PPT's Latent Traversal)
  factor_traversal.png   traversal along the best dimension for rotation / thickness / class
  factor_importance.png  factor probe importance heatmap (per-dimension, per-factor)
  latent_space.png       PCA + t-SNE visualization of the latent space (colored by class)
  training_curves.png    training curves
  metrics.json           quantitative metrics (MIG, probe R^2 / accuracy, top dim per factor)
"""
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import config
from dataset import get_dataloaders, load_clean_mnist
from model import BetaVAE
from metrics import mig, linear_probe_regression, linear_probe_classification

FACTOR_NAMES = ["rotation", "thickness", "class"]


def load_model(device):
    ckpt = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
    model = BetaVAE(ckpt["latent_dim"], config.HIDDEN_DIMS).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def encode(model, x, device, chunk=512):
    """Batch-encode, returning the posterior mean mu of shape [N, D]."""
    mus = []
    for i in range(0, x.size(0), chunk):
        mu, _ = model.encoder(x[i:i + chunk].to(device))
        mus.append(mu.cpu())
    return torch.cat(mus, 0).numpy()


def make_grid(imgs, nrow):
    """[N, 1, H, W] -> a single large image of shape [H * rows, W * nrow]."""
    N = imgs.shape[0]
    H, W = imgs.shape[2], imgs.shape[3]
    rows = (N + nrow - 1) // nrow
    grid = np.zeros((rows * H, nrow * W), dtype=np.float32)
    for i in range(N):
        r, c = divmod(i, nrow)
        grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = imgs[i, 0]
    return grid


def traverse(model, mu, dim, n_steps=11, z_range=(-3, 3)):
    """Fix all other dims and vary only `dim`; return the image grid and the traversal values."""
    steps = np.linspace(*z_range, n_steps)
    z = mu.unsqueeze(0).repeat(n_steps, 1)
    z[:, dim] = torch.tensor(steps, dtype=z.dtype, device=z.device)
    with torch.no_grad():
        out = model.decoder(z)
    return make_grid(out.cpu().numpy(), n_steps), steps


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_model(device)
    beta = float(ckpt.get("beta", config.BETA))

    # ------------------------------------------------------------------
    # 1) Clean test images + reference image (digit 8: more strokes, so traversal is more intuitive)
    # ------------------------------------------------------------------
    clean, clean_labels = load_clean_mnist(config.DATA_DIR, train=False)
    ref_idx = int((clean_labels == 8).nonzero()[0])
    ref = clean[ref_idx:ref_idx + 1]  # [1, 1, 28, 28]

    # ------------------------------------------------------------------
    # 2) Reconstruction comparison
    # ------------------------------------------------------------------
    with torch.no_grad():
        x8 = clean[:8].to(device)
        x8r, _, _ = model(x8)
    orig = make_grid(x8.cpu().numpy(), 8)
    recon = make_grid(x8r.cpu().numpy(), 8)
    fig, axes = plt.subplots(2, 1, figsize=(5, 2.0))
    for ax, g, t in zip(axes, [orig, recon], ["Original", "Reconstruction"]):
        ax.imshow(g, cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(t, fontsize=9)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "reconstructions.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3) Encode the test set into latent codes + collect factors
    # ------------------------------------------------------------------
    _, test_loader = get_dataloaders(config)
    xs, ys, rots, thks = [], [], [], []
    for x, y, rot, thk in test_loader:
        xs.append(x); ys.append(y); rots.append(rot); thks.append(thk)
    x_all = torch.cat(xs, 0)
    y = torch.cat(ys, 0).numpy()
    rot = torch.cat(rots, 0).numpy()
    thk = torch.cat(thks, 0).numpy()
    print("[evaluate] encoding test set ...")
    z = encode(model, x_all, device)

    # ------------------------------------------------------------------
    # 4) Factor probes + MIG
    # ------------------------------------------------------------------
    r2_rot, imp_rot = linear_probe_regression(z, rot)
    r2_thk, imp_thk = linear_probe_regression(z, thk)
    acc_cls, imp_cls = linear_probe_classification(z, y)
    mig_cls, _, _ = mig(z, y)
    mig_thk, _, _ = mig(z, thk - 1)  # thickness 1..3 -> 0..2

    top = {"rotation": int(np.argmax(imp_rot)),
           "thickness": int(np.argmax(imp_thk)),
           "class": int(np.argmax(imp_cls))}

    imp = np.stack([imp_rot, imp_thk, imp_cls], 0)          # [3, D]
    imp_norm = imp / (imp.max(1, keepdims=True) + 1e-9)
    dom = imp_norm.argmax(0)                                # dominant factor per dimension

    metrics = {
        "latent_dim": int(model.latent_dim),
        "beta": beta,
        "mig_class": round(mig_cls, 4),
        "mig_thickness": round(mig_thk, 4),
        "probe_rotation_r2": round(r2_rot, 4),
        "probe_thickness_r2": round(r2_thk, 4),
        "probe_class_accuracy": round(acc_cls, 4),
        "top_dim_rotation": top["rotation"],
        "top_dim_thickness": top["thickness"],
        "top_dim_class": top["class"],
    }

    # ------------------------------------------------------------------
    # 5) Latent traversal (core figure)
    # ------------------------------------------------------------------
    with torch.no_grad():
        mu_ref, _ = model.encoder(ref.to(device))
    mu_ref = mu_ref.squeeze(0)
    D = model.latent_dim
    n_steps = 11
    rows = [traverse(model, mu_ref, d, n_steps)[0] for d in range(D)]
    H, W = rows[0].shape
    big = np.vstack(rows)
    fig, ax = plt.subplots(figsize=(n_steps * 0.5, D * 0.55))
    ax.imshow(big, cmap="gray_r", vmin=0, vmax=1)
    for d in range(D):
        ax.text(-W * 0.45, d * H + H / 2, f"z{d} ({FACTOR_NAMES[dom[d]]})",
                va="center", ha="right", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-W * 2.4, W * n_steps)
    ax.set_title(f"Latent Traversal (beta={beta})", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "latent_traversal.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6) Factor traversal (the best dim for rotation / thickness / class)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(n_steps * 0.5, 3 * 0.6))
    for ax, name in zip(axes, FACTOR_NAMES):
        g, _ = traverse(model, mu_ref, top[name], n_steps)
        ax.imshow(g, cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"dim z{top[name]} -> {name}", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "factor_traversal.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 7) Factor importance heatmap
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, D * 0.4), 3))
    im = ax.imshow(imp_norm, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(3))
    ax.set_yticklabels(FACTOR_NAMES)
    ax.set_xticks(range(D))
    ax.set_xticklabels([f"z{d}" for d in range(D)], rotation=90, fontsize=7)
    ax.set_xlabel("latent dimension")
    ax.set_title("Factor Probe Importance (normalized)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "factor_importance.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8) Latent space visualization (PCA + t-SNE, colored by class)
    # ------------------------------------------------------------------
    subsample = min(5000, z.shape[0])
    idx = np.random.RandomState(0).choice(z.shape[0], subsample, replace=False)
    zs, ys = z[idx], y[idx]
    pca = PCA(n_components=2).fit_transform(zs)
    tsne = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(zs)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    sc1 = axes[0].scatter(pca[:, 0], pca[:, 1], c=ys, cmap="tab10", s=4)
    axes[0].set_title("PCA of latent space")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    fig.colorbar(sc1, ax=axes[0], ticks=range(10))
    sc2 = axes[1].scatter(tsne[:, 0], tsne[:, 1], c=ys, cmap="tab10", s=4)
    axes[1].set_title("t-SNE of latent space")
    axes[1].set_xlabel("t-SNE1"); axes[1].set_ylabel("t-SNE2")
    fig.colorbar(sc2, ax=axes[1], ticks=range(10))
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "latent_space.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 9) Training curves
    # ------------------------------------------------------------------
    if config.HISTORY_PATH.exists():
        hist = json.loads(config.HISTORY_PATH.read_text(encoding="utf-8"))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        for ax, key in zip(axes, ["loss", "recon", "kl"]):
            ax.plot(hist["epoch"], hist[key])
            ax.set_title(key)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(config.OUTPUT_DIR / "training_curves.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 10) Save metrics
    # ------------------------------------------------------------------
    (config.OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[evaluate] results saved to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
