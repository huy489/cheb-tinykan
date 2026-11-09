"""
Cheb-TinyKAN: End-to-End Training & Inter-patient Benchmark Script.
Designed to run on NVIDIA T4 GPU (Google Colab) or local multi-core environments.

Features:
- Inter-patient 10-fold PTB-XL split (Train: Folds 1-8, Val: Fold 9, Test: Fold 10).
- Multi-label Focal Loss (gamma=2.0, alpha=0.25) with L1 Sparsification penalty.
- Metrics tracking: Macro-AUC, Macro-F1, Sensitivity/Specificity per territory.
- Checkpoint persistence and training history JSON export.
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR
except ImportError:
    torch = None

from models.model import ChebTinyKAN
from losses.focal_loss import MultiLabelFocalLoss
from data.ptbxl_dataset import create_dataloaders
from data.scp_mapping import TARGET_CLASSES


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """
    Computes Macro-F1, Sensitivity, Specificity, and per-head metrics without external sklearn.
    Args:
        y_true: (N, 5) binary ground truth
        y_prob: (N, 5) float probabilities in [0, 1]
    """
    y_pred = (y_prob >= threshold).astype(np.float32)
    num_classes = y_true.shape[1]

    f1_scores = []
    sensitivities = []
    specificities = []

    per_class = {}

    for c in range(num_classes):
        tp = np.sum((y_pred[:, c] == 1) & (y_true[:, c] == 1))
        fp = np.sum((y_pred[:, c] == 1) & (y_true[:, c] == 0))
        fn = np.sum((y_pred[:, c] == 0) & (y_true[:, c] == 1))
        tn = np.sum((y_pred[:, c] == 0) & (y_true[:, c] == 0))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        spec = tn / (tn + fp + 1e-8)

        f1_scores.append(f1)
        sensitivities.append(recall)
        specificities.append(spec)

        per_class[TARGET_CLASSES[c]] = {
            "f1": float(f1),
            "sensitivity": float(recall),
            "specificity": float(spec),
            "support": int(np.sum(y_true[:, c]))
        }

    return {
        "macro_f1": float(np.mean(f1_scores)),
        "macro_sensitivity": float(np.mean(sensitivities)),
        "macro_specificity": float(np.mean(specificities)),
        "per_class": per_class
    }


def train_epoch(model, loader, optimizer, criterion, l1_lambda, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)
        focal_loss = criterion(logits, y)
        l1_loss = l1_lambda * model.l1_regularization_loss()
        loss = focal_loss + l1_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / max(1, total_samples)


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = model.predict_probabilities(x)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = calculate_metrics(all_targets, all_probs)
    return metrics, all_probs, all_targets


def main():
    parser = argparse.ArgumentParser(description="Cheb-TinyKAN Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="./data/ptb-xl", help="Path to PTB-XL root")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--l1_lambda", type=float, default=1e-4, help="L1 sparsity penalty weight")
    parser.add_argument("--device", type=str, default="cuda" if torch and torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if torch is None:
        print("ERROR: PyTorch is not installed in current environment. Please run on Colab or environment with PyTorch.")
        sys.exit(1)

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 70)
    print("🚀 CHEB-TINYKAN: ORTHOGONAL KAN FOR EDGE ECG ARRHYTHMIA & MI LOCALIZATION")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size} | L1: {args.l1_lambda}")
    print(f"Data directory: {args.data_dir}")
    print("=" * 70)

    # 1. Create Loaders
    print("Loading PTB-XL Inter-patient split datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=2 if device.type == 'cuda' else 0
    )
    print(f"Train samples (Folds 1-8): {len(train_loader.dataset)}")
    print(f"Val samples   (Fold 9):    {len(val_loader.dataset)}")
    print(f"Test samples  (Fold 10):   {len(test_loader.dataset)}")

    # 2. Instantiate Model
    model = ChebTinyKAN(in_channels=12, num_classes=5, degree=4).to(device)
    budget = model.count_parameters()
    print("\n📦 TinyML Budget Audit:")
    print(f"  - 1D-CNN Tokenizer params: {budget['tokenizer_params']:,}")
    print(f"  - Cheb-KAN Bottleneck:     {budget['bottleneck_kan_params']:,}")
    print(f"  - Cheb-KAN Decision Head:  {budget['decision_kan_params']:,}")
    print(f"  - TOTAL PARAMETERS:        {budget['total_params']:,} (Target < 20K)")
    print("-" * 70)

    # 3. Optimizer, Scheduler, Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = MultiLabelFocalLoss(alpha=0.25, gamma=2.0)

    best_val_f1 = 0.0
    best_epoch = 0
    history = []

    # 4. Training Loop
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.l1_lambda, device)
        scheduler.step()

        val_metrics, _, _ = evaluate(model, val_loader, device)
        val_f1 = val_metrics["macro_f1"]
        ep_time = time.time() - ep_start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": val_f1,
            "val_per_class": val_metrics["per_class"],
            "lr": optimizer.param_groups[0]["lr"]
        })

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "config": vars(args)
            }, os.path.join(args.save_dir, "model_best.pth"))

        marker = "🔥 BEST" if is_best else ""
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {train_loss:.4f} | Val F1: {val_f1*100:.2f}% | Time: {ep_time:.1f}s {marker}")

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"🎉 Training Completed in {total_time/60:.2f} mins! Best Val Macro-F1: {best_val_f1*100:.2f}% (Epoch {best_epoch})")

    # 5. Final Evaluation on Strict Inter-patient Test Set (Fold 10)
    print("\n🔬 Final Inter-patient Testing on Fold 10 (Zero-Leakage Benchmark):")
    checkpoint = torch.load(os.path.join(args.save_dir, "model_best.pth"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, test_probs, test_targets = evaluate(model, test_loader, device)
    print(f"  >> Test Set Macro-F1:          {test_metrics['macro_f1']*100:.2f}%")
    print(f"  >> Test Set Macro-Sensitivity: {test_metrics['macro_sensitivity']*100:.2f}%")
    print(f"  >> Test Set Macro-Specificity: {test_metrics['macro_specificity']*100:.2f}%")

    print("\n📊 Per-Territory Clinical Metrics:")
    for cls_name, m in test_metrics["per_class"].items():
        print(f"  - {cls_name:18s}: F1 = {m['f1']*100:5.2f}% | Sens = {m['sensitivity']*100:5.2f}% | Spec = {m['specificity']*100:5.2f}% (N={m['support']})")

    # 6. Sparsification & Pruning Audit
    pruned_edges = model.bottleneck_kan.prune_edges(threshold=1e-3)
    total_edges = 64 * 24
    print(f"\n✂️ L1 Sparsity Pruning Audit: {pruned_edges}/{total_edges} edges pruned ({pruned_edges/total_edges*100:.1f}%)")

    # 7. Save Metrics Summary
    output_summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "budget": budget,
        "pruned_edges_ratio": pruned_edges / total_edges,
        "training_time_seconds": total_time,
        "history": history
    }
    with open(os.path.join(args.save_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)

    print(f"\n📁 Saved deliverables to: {args.save_dir}/")
    print(f"  1. Best Weights:   {os.path.join(args.save_dir, 'model_best.pth')}")
    print(f"  2. Metrics Report: {os.path.join(args.save_dir, 'training_metrics.json')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
