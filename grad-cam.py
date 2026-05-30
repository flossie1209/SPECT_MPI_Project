#!/usr/bin/env python3
"""
Grad-CAM Pipeline for SPECT MPI Binary Classification
======================================================
Normal vs Abnormal (Ischemia / Obstructive CAD)

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) from scratch:
  1. Register forward/backward hooks on the last ResNet conv block (layer4[-1])
  2. Forward pass  →  feature maps A^k  (shape: 1 × K × H' × W')
  3. Backward pass →  gradients ∂y^c / ∂A^k_ij
  4. Global average pool gradients → per-channel weights (α^c_k)
  5. Weighted sum + ReLU  →  localization map L^c
  6. Upsample & normalize to [0,1], overlay on original image  →  CAD attention map

Usage:
    python gradcam_spect_mpi.py

Expected folder layout (auto-created by the script with provided zip dataset):
    data/
    ├── train/  Abnormal/  Normal/
    ├── val/    Abnormal/  Normal/
    └── test/   Abnormal/  Normal/
    results_gradcam/
        gradcam_abnormal.png
        gradcam_normal.png
        gradcam_comparison.png
        cad_probability_map.png
        training_curves.png
        confusion_matrix.png
        individual_heatmaps/
"""

import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
from tqdm import tqdm

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DATA_DIR       = Path("data")          # pre-split train / val / test
RESULTS_DIR    = Path("results_gradcam")
IMG_SIZE       = 224
BATCH_SIZE     = 8
EPOCHS         = 20
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
SEED           = 42
USE_PRETRAINED = True    # set False if PyTorch Hub is not reachable

IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
# Reproducibility / device
# ─────────────────────────────────────────────

def set_seed(s: int = 42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device():
    if torch.cuda.is_available():      return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────

def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


# ─────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────

def build_loaders():
    train_tf, eval_tf = build_transforms()
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(DATA_DIR / "val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(DATA_DIR / "test",  transform=eval_tf)

    train_ldr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_ldr   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_ldr  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_names = train_ds.classes
    print(f"Classes : {class_names}  →  {train_ds.class_to_idx}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    return train_ldr, val_ldr, test_ldr, test_ds, class_names, train_ds


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

def build_model(num_classes: int, device):
    weights = models.ResNet18_Weights.DEFAULT if USE_PRETRAINED else None
    model   = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


# ─────────────────────────────────────────────
# Train / eval helpers
# ─────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0; all_p = []; all_l = []
    for imgs, labs in tqdm(loader, desc="  train", leave=False):
        imgs, labs = imgs.to(device), labs.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labs)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        all_p.extend(out.argmax(1).cpu().numpy())
        all_l.extend(labs.cpu().numpy())
    return total_loss / len(loader.dataset), accuracy_score(all_l, all_p)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, desc="  val"):
    model.eval()
    total_loss = 0.0; all_p = []; all_l = []
    for imgs, labs in tqdm(loader, desc=desc, leave=False):
        imgs, labs = imgs.to(device), labs.to(device)
        out  = model(imgs)
        loss = criterion(out, labs)
        total_loss += loss.item() * imgs.size(0)
        all_p.extend(out.argmax(1).cpu().numpy())
        all_l.extend(labs.cpu().numpy())
    return total_loss / len(loader.dataset), accuracy_score(all_l, all_p), \
           np.array(all_l), np.array(all_p)


# ─────────────────────────────────────────────
# ██  GRAD-CAM  (from scratch — no external lib)
# ─────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.

    Mathematical formulation
    ------------------------
    Forward:
        A^k  =  activation of channel k at target convolutional layer
                shape: [K, H', W']

    Backward:
        α^c_k  =  (1/Z) ΣΣ_ij  ∂y^c / ∂A^k_ij
                = global-average-pool of gradient w.r.t. A^k

    Localization map:
        L^c_Grad-CAM  =  ReLU( Σ_k  α^c_k · A^k )

        ReLU keeps only regions that *increase* the target class score.

    Final heat map:
        Upsample L^c to input size, normalize to [0, 1].
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model  = model
        self.fmaps  = None      # A^k filled by forward hook
        self.grads  = None      # ∂y^c/∂A^k filled by backward hook

        self._fh = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "fmaps", o)
        )
        self._bh = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "grads", go[0])
        )

    def generate(
        self,
        img_tensor: torch.Tensor,
        target_class: int = None,
    ):
        """
        Parameters
        ----------
        img_tensor   : C×H×W tensor (no batch dimension)
        target_class : class index to explain; defaults to predicted class

        Returns
        -------
        cam           : H×W numpy array in [0, 1]  (localization map)
        pred_class    : int  — predicted class index
        pred_prob     : float — predicted class probability
        """
        self.model.eval()
        x = img_tensor.unsqueeze(0)           # [1, C, H, W]

        # ── 1. Forward pass ──────────────────────────────────────────
        logits = self.model(x)                # [1, num_classes]
        probs  = torch.softmax(logits, dim=1)
        pred   = int(logits.argmax(1).item())
        prob   = float(probs[0, pred].item())

        if target_class is None:
            target_class = pred

        # ── 2. Backward pass ─────────────────────────────────────────
        self.model.zero_grad()
        logits[0, target_class].backward(retain_graph=True)

        # ── 3. α^c_k = global average pool of gradients ─────────────
        alpha = self.grads.squeeze(0).mean(dim=[1, 2])       # [K]

        # ── 4. Weighted sum of feature maps ──────────────────────────
        fmaps = self.fmaps.squeeze(0)                        # [K, H', W']
        cam   = (alpha[:, None, None] * fmaps).sum(0)        # [H', W']

        # ── 5. ReLU (positive influence only) ────────────────────────
        cam = F.relu(cam)

        # ── 6. Upsample + normalize ───────────────────────────────────
        H, W = img_tensor.shape[1], img_tensor.shape[2]
        cam = F.interpolate(
            cam[None, None], size=(H, W), mode="bilinear", align_corners=False
        ).squeeze().detach().numpy()

        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo + 1e-8)

        return cam, pred, prob

    def remove_hooks(self):
        self._fh.remove()
        self._bh.remove()


# ─────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────

def unnormalize(tensor: torch.Tensor) -> np.ndarray:
    """Tensor [C,H,W] → RGB numpy [H,W,C] in [0,1]."""
    mean = np.array(IMAGENET_MEAN)
    std  = np.array(IMAGENET_STD)
    img  = tensor.permute(1, 2, 0).numpy()
    return np.clip(std * img + mean, 0, 1)


def to_heatmap(cam: np.ndarray) -> np.ndarray:
    """[H,W] in [0,1]  →  jet colormap [H,W,3]."""
    return matplotlib.colormaps["jet"](cam)[:, :, :3]


def superimpose(orig: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend Grad-CAM heatmap over original image."""
    return np.clip((1 - alpha) * orig + alpha * to_heatmap(cam), 0, 1)


def _add_colorbar(fig, ax, label="Activation intensity"):
    sm = ScalarMappable(cmap="jet", norm=Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=7)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Low", "Mid", "High"])


def plot_cam_grid(examples, class_names, save_path, title, max_rows=6):
    """
    3-column grid: Original | Grad-CAM heatmap | CAD attention map overlay
    """
    examples = examples[:max_rows]
    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.2 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.005)

    for row, (img_t, cam, true_idx, pred_idx, prob, _) in enumerate(examples):
        orig = unnormalize(img_t)
        heat = to_heatmap(cam)
        sup  = superimpose(orig, cam)
        tl, pl, ok = class_names[true_idx], class_names[pred_idx], ("✓" if true_idx == pred_idx else "✗")

        for col, (dat, sub) in enumerate([
            (orig, "Original SPECT MPI Polar Map"),
            (heat, "Grad-CAM Heatmap\n(Red = peak CAD activation)"),
            (sup,  f"CAD Attention Map  {ok}\nPred: {pl} ({prob:.0%})  |  True: {tl}"),
        ]):
            ax = axes[row][col]
            ax.imshow(dat); ax.set_title(sub, fontsize=8.5); ax.axis("off")

        _add_colorbar(fig, axes[row][1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_probability_map(examples, class_names, abnormal_idx, save_path):
    """
    CAD Probability Map: Grad-CAM scaled by prediction confidence.
    Mirrors the 'CAD probability map' concept in the CAD-DL paper.
    """
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5))
    if n == 1: axes = [axes]
    fig.suptitle(
        "CAD Probability Maps  (Grad-CAM × Prediction Confidence)",
        fontsize=11, fontweight="bold",
    )
    for ax, (img_t, cam, true_idx, pred_idx, prob, _) in zip(axes, examples):
        # Scale activation by probability of CAD
        cad_prob_cam = cam * (prob if pred_idx == abnormal_idx else 1 - prob)
        sup = superimpose(unnormalize(img_t), cad_prob_cam, alpha=0.5)
        ax.imshow(sup); ax.axis("off")
        ax.set_title(
            f"True: {class_names[true_idx]}\nPred: {class_names[pred_idx]} ({prob:.0%})",
            fontsize=8.5,
        )
    sm = ScalarMappable(cmap="jet", norm=Normalize(0, 1)); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[-1], fraction=0.046, pad=0.04)
    cb.set_label("CAD probability", fontsize=8)
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["Low", "Med", "High"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# Metrics plots
# ─────────────────────────────────────────────

def plot_training_curves(history, results_dir):
    epochs = range(1, len(history["tl"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["tl"], label="Train"); axes[0].plot(epochs, history["vl"], label="Val")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss"); axes[0].legend()
    axes[1].plot(epochs, history["ta"], label="Train"); axes[1].plot(epochs, history["va"], label="Val")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150); plt.close()


def plot_confusion_matrix(labels, preds, class_names, results_dir):
    cm   = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, xticks_rotation=30, values_format="d")
    ax.set_title("Confusion Matrix — SPECT MPI CNN")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150); plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_ldr, val_ldr, test_ldr, test_ds, class_names, train_ds = build_loaders()
    num_classes  = len(class_names)
    abnormal_idx = class_names.index("Abnormal") if "Abnormal" in class_names else 0

    # Class weights (compensate for imbalance)
    counts  = np.bincount([s[1] for s in train_ds.samples])
    w       = 1.0 / counts.astype(float); w = w / w.sum() * num_classes
    cw      = torch.tensor(w, dtype=torch.float).to(device)
    print(f"Class weights: {dict(zip(class_names, w.round(3)))}")

    # ── Model, loss, optimizer ────────────────────────────────────────────
    model     = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = -np.inf
    best_sd      = None
    history      = {"tl": [], "ta": [], "vl": [], "va": []}

    # ── Training loop ────────────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs…")
    for ep in range(1, EPOCHS + 1):
        tl, ta = train_epoch(model, train_ldr, criterion, optimizer, device)
        vl, va, _, _ = eval_epoch(model, val_ldr, criterion, device)
        scheduler.step()

        history["tl"].append(tl); history["ta"].append(ta)
        history["vl"].append(vl); history["va"].append(va)
        print(f"Ep {ep:>2}/{EPOCHS}  tl={tl:.4f}  ta={ta:.4f}  vl={vl:.4f}  va={va:.4f}")

        if va >= best_val_acc:
            best_val_acc = va
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"Best val accuracy: {best_val_acc:.4f}")
    model.load_state_dict(best_sd)
    torch.save(best_sd, RESULTS_DIR / "best_model.pth")

    plot_training_curves(history, RESULTS_DIR)

    # ── Test evaluation ───────────────────────────────────────────────────
    _, te_acc, te_labels, te_preds = eval_epoch(model, test_ldr, criterion, device, desc="  test")
    print(f"Test accuracy: {te_acc:.4f}")
    print(classification_report(te_labels, te_preds, target_names=class_names, digits=4, zero_division=0))
    plot_confusion_matrix(te_labels, te_preds, class_names, RESULTS_DIR)

    # ─────────────────────────────────────────────────────────────────────
    # ██  GRAD-CAM  generation
    # ─────────────────────────────────────────────────────────────────────
    print("\nGenerating Grad-CAM attention maps…")
    gc = GradCAM(model, target_layer=model.layer4[-1])

    all_ex = []
    for idx, (path, true_lbl) in enumerate(test_ds.samples):
        img_t, _ = test_ds[idx]
        cam, pred, prob = gc.generate(img_t.to(device))
        all_ex.append((img_t, cam, true_lbl, pred, prob, Path(path).stem))

    gc.remove_hooks()

    abn_ex = [e for e in all_ex if e[2] == abnormal_idx]
    nrm_ex = [e for e in all_ex if e[2] != abnormal_idx]

    # 1. Abnormal attention maps
    if abn_ex:
        plot_cam_grid(
            abn_ex[:6], class_names,
            RESULTS_DIR / "gradcam_abnormal.png",
            title="Grad-CAM — Abnormal SPECT MPI  (Ischemia / CAD Regions Highlighted)",
        )

    # 2. Normal reference maps
    if nrm_ex:
        plot_cam_grid(
            nrm_ex, class_names,
            RESULTS_DIR / "gradcam_normal.png",
            title="Grad-CAM — Normal SPECT MPI  (Reference Activations)",
        )

    # 3. Side-by-side comparison
    comp = nrm_ex[:2] + abn_ex[:2]
    if comp:
        plot_cam_grid(
            comp, class_names,
            RESULTS_DIR / "gradcam_comparison.png",
            title="Grad-CAM — Normal vs Abnormal Comparison",
        )

    # 4. CAD probability maps
    prob_exs = nrm_ex[:2] + abn_ex[:3]
    if prob_exs:
        plot_probability_map(prob_exs, class_names, abnormal_idx, RESULTS_DIR / "cad_probability_map.png")

    # 5. Individual heatmaps for all test images
    ind_dir = RESULTS_DIR / "individual_heatmaps"; ind_dir.mkdir(exist_ok=True)
    for img_t, cam, true_idx, pred_idx, prob, stem in all_ex:
        orig = unnormalize(img_t); heat = to_heatmap(cam); sup = superimpose(orig, cam)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        axes[0].imshow(orig);  axes[0].set_title("Original SPECT MPI");    axes[0].axis("off")
        axes[1].imshow(heat);  axes[1].set_title("Grad-CAM Heatmap");      axes[1].axis("off")
        axes[2].imshow(sup);   axes[2].set_title(
            f"CAD Attention Map\nPred: {class_names[pred_idx]} ({prob:.0%})  True: {class_names[true_idx]}"
        ); axes[2].axis("off")
        _add_colorbar(fig, axes[1])
        plt.tight_layout()
        plt.savefig(ind_dir / f"{stem}_gradcam.png", dpi=120, bbox_inches="tight")
        plt.close()

    print(f"  Saved {len(all_ex)} individual heatmaps → {ind_dir}")

    print(f"\n✓ All results saved in: {RESULTS_DIR.resolve()}")
    print("  gradcam_abnormal.png     — ischemia attention maps")
    print("  gradcam_normal.png       — normal reference maps")
    print("  gradcam_comparison.png   — side-by-side comparison")
    print("  cad_probability_map.png  — CAD probability overlays")
    print("  training_curves.png      — loss / accuracy curves")
    print("  confusion_matrix.png     — test-set confusion matrix")
    print("  individual_heatmaps/     — per-image Grad-CAM outputs")


if __name__ == "__main__":
    main()
