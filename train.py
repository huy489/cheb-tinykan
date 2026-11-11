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


def compute_roc_auc(y_true_col: np.ndarray, y_prob_col: np.ndarray) -> float:
    """Exact Wilcoxon-Mann-Whitney ROC-AUC with tie handling in pure NumPy."""
    pos = (y_true_col == 1)
    n_pos = int(np.sum(pos))
    n_neg = int(len(y_true_col) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_prob_col, kind="mergesort")
    sorted_probs = y_prob_col[order]
    sorted_pos = pos[order]
    ranks = np.arange(1, len(y_prob_col) + 1, dtype=np.float64)
    _, idx, counts = np.unique(sorted_probs, return_index=True, return_counts=True)
    for start, cnt in zip(idx, counts):
        if cnt > 1:
            ranks[start:start + cnt] = np.mean(ranks[start:start + cnt])
    rank_sum_pos = np.sum(ranks[sorted_pos])
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold=0.5):
    """
    Computes Macro-F1, Macro-AUC, Sensitivity, Specificity, and per-head metrics without external sklearn.
    Args:
        y_true: (N, 5) binary ground truth
        y_prob: (N, 5) float probabilities in [0, 1]
        threshold: scalar float or sequence of length num_classes
    """
    thresh_arr = np.asarray(threshold, dtype=np.float32)
    if thresh_arr.ndim == 0:
        thresh_arr = np.full(y_true.shape[1], float(thresh_arr), dtype=np.float32)

    y_pred = (y_prob >= thresh_arr).astype(np.float32)
    num_classes = y_true.shape[1]

    f1_scores = []
    auc_scores = []
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
        auc = compute_roc_auc(y_true[:, c], y_prob[:, c])

        f1_scores.append(f1)
        auc_scores.append(auc)
        sensitivities.append(recall)
        specificities.append(spec)

        per_class[TARGET_CLASSES[c]] = {
            "f1": float(f1),
            "auc": float(auc),
            "sensitivity": float(recall),
            "specificity": float(spec),
            "threshold": float(thresh_arr[c]),
            "support": int(np.sum(y_true[:, c]))
        }

    return {
        "macro_f1": float(np.mean(f1_scores)),
        "macro_auc": float(np.mean(auc_scores)),
        "macro_sensitivity": float(np.mean(sensitivities)),
        "macro_specificity": float(np.mean(specificities)),
        "per_class": per_class
    }


def optimize_thresholds_on_val(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """
    Searches optimal per-class probability threshold strictly on Validation Fold 9
    maximizing F1 per class (never touching Test Fold 10).
    """
    num_classes = y_true.shape[1]
    best_thresh = np.full(num_classes, 0.5, dtype=np.float32)
    grid = np.linspace(0.15, 0.85, 71)

    for c in range(num_classes):
        best_f1 = -1.0
        for th in grid:
            pred = (y_prob[:, c] >= th).astype(np.float32)
            tp = np.sum((pred == 1) & (y_true[:, c] == 1))
            fp = np.sum((pred == 1) & (y_true[:, c] == 0))
            fn = np.sum((pred == 0) & (y_true[:, c] == 1))
            f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh[c] = float(th)
    return best_thresh


def compute_balanced_alpha_from_dataset(train_dataset) -> list:
    """
    Computes inverse square-root class balance weights:
        alpha_c = sqrt(N_neg / N_pos) / (1 + sqrt(N_neg / N_pos))
    exclusively from Train Folds 1-8 labels.
    """
    if hasattr(train_dataset, "labels"):
        y_all = train_dataset.labels
    else:
        y_all = np.array([s["label"] for s in train_dataset.samples], dtype=np.float32)

    alphas = []
    n_total = len(y_all)
    for c in range(y_all.shape[1]):
        n_pos = max(1.0, float(np.sum(y_all[:, c])))
        n_neg = max(1.0, float(n_total - n_pos))
        ratio_sqrt = np.sqrt(n_neg / n_pos)
        alpha_c = float(ratio_sqrt / (1.0 + ratio_sqrt))
        alphas.append(round(alpha_c, 4))
    return alphas


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


def evaluate(model, loader, device, thresholds=0.5):
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = model.predict_probabilities(x)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = calculate_metrics(all_targets, all_probs, threshold=thresholds)
    return metrics, all_probs, all_targets


def main():
    parser = argparse.ArgumentParser(description="Cheb-TinyKAN Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="./data/ptb-xl", help="Path to PTB-XL root")
    parser.add_argument("--cache_path", type=str, default=None, help="Optional path to preprocessed .npz cache")
    parser.add_argument("--arch", type=str, default="baseline", choices=["baseline", "temporal_zones"], help="Tokenizer architecture")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--l1_lambda", type=float, default=1e-4, help="Group-L1 edge sparsity penalty weight")
    parser.add_argument("--alpha", type=float, default=0.25, help="Scalar Focal loss alpha (ignored if --auto_alpha set)")
    parser.add_argument("--auto_alpha", action="store_true", help="Derive per-class balanced alpha from Train Folds 1-8")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma focusing parameter")
    parser.add_argument("--prune_epoch", type=int, default=0, help="Epoch to lock target sparsity pruning in bottleneck KAN (0=disabled)")
    parser.add_argument("--target_sparsity", type=float, default=0.70, help="Fraction of Bottleneck KAN edges to prune")
    parser.add_argument("--device", type=str, default="cuda" if torch and torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_last.pth if present")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if torch is None:
        print("ERROR: PyTorch is not installed in current environment. Please run on Colab or environment with PyTorch.")
        sys.exit(1)

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    # 1. Create Loaders
    print("=" * 70)
    print("🚀 CHEB-TINYKAN: ORTHOGONAL KAN FOR EDGE ECG ARRHYTHMIA & MI LOCALIZATION")
    print(f"Arch: {args.arch} | Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size} | L1: {args.l1_lambda}")
    print("=" * 70)
    print("Loading PTB-XL Inter-patient split datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=2 if device.type == 'cuda' else 0,
        cache_path=args.cache_path
    )
    print(f"Train samples (Folds 1-8): {len(train_loader.dataset)}")
    print(f"Val samples   (Fold 9):    {len(val_loader.dataset)}")
    print(f"Test samples  (Fold 10):   {len(test_loader.dataset)}")

    if args.auto_alpha:
        alpha_val = compute_balanced_alpha_from_dataset(train_loader.dataset)
        print(f"⚖️ Auto-Balanced Alpha (derived from Train Folds 1-8): {dict(zip(TARGET_CLASSES, alpha_val))}")
    else:
        alpha_val = args.alpha
        print(f"⚖️ Scalar Alpha: {alpha_val}")

    # 2. Instantiate Model
    model = ChebTinyKAN(in_channels=12, num_classes=5, degree=4, arch=args.arch).to(device)
    budget = model.count_parameters()
    print("\n📦 TinyML Budget Audit:")
    print(f"  - 1D-CNN Tokenizer params: {budget['tokenizer_params']:,}")
    print(f"  - Cheb-KAN Bottleneck:     {budget['bottleneck_kan_params']:,}")
    print(f"  - Cheb-KAN Decision Head:  {budget['decision_kan_params']:,}")
    print(f"  - TOTAL RAW PARAMETERS:    {budget['total_params']:,}")
    print("-" * 70)

    # 3. Optimizer, Scheduler, Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = MultiLabelFocalLoss(alpha=alpha_val, gamma=args.gamma).to(device)

    start_epoch = 1
    best_val_f1 = 0.0
    best_epoch = 0
    history = []

    last_ckpt_path = os.path.join(args.save_dir, "checkpoint_last.pth")
    if args.resume and os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        best_epoch = ckpt.get("best_epoch", 0)
        history = ckpt.get("history", [])
        print(f"♻️ Resuming from epoch {start_epoch} (Best Val F1 so far: {best_val_f1*100:.2f}% at Ep {best_epoch})")

    # 4. Training Loop
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        ep_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.l1_lambda, device)
        scheduler.step()

        # Apply structured pruning & reset best tracker so best model reflects final sparse weights
        if args.prune_epoch > 0 and epoch == args.prune_epoch:
            pruned_cnt = model.bottleneck_kan.prune_by_sparsity(target_sparsity=args.target_sparsity)
            active_budget = model.count_parameters()
            print(f"\n✂️ [Epoch {epoch}] Pruned {pruned_cnt}/1536 Bottleneck edges ({pruned_cnt/1536*100:.1f}%) -> Active Params: {active_budget['active_total_params']:,} (< 12,000: {active_budget['within_budget']})")
            best_val_f1 = 0.0  # Force best selection among pruned sparse fine-tuning epochs

        val_metrics, _, _ = evaluate(model, val_loader, device)
        val_f1 = val_metrics["macro_f1"]
        val_auc = val_metrics["macro_auc"]
        ep_time = time.time() - ep_start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": val_f1,
            "val_macro_auc": val_auc,
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
                "val_macro_auc": val_auc,
                "arch": args.arch,
                "alpha": alpha_val,
                "config": vars(args)
            }, os.path.join(args.save_dir, "model_best.pth"))

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "history": history,
            "config": vars(args)
        }, last_ckpt_path)

        marker = "🔥 BEST" if is_best else ""
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {train_loss:.4f} | Val F1@0.5: {val_f1*100:.2f}% | Val AUC: {val_auc:.4f} | Time: {ep_time:.1f}s {marker}")

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"🎉 Training Completed in {total_time/60:.2f} mins! Best Val Macro-F1@0.5: {best_val_f1*100:.2f}% (Epoch {best_epoch})")

    # 5. Strict Validation-Only Threshold Optimization -> Frozen Test on Fold 10
    print("\n🔍 Loading Best Model for Strict Inter-patient Evaluation:")
    checkpoint = torch.load(os.path.join(args.save_dir, "model_best.pth"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics_05, val_probs, val_targets = evaluate(model, val_loader, device, thresholds=0.5)
    opt_thresholds = optimize_thresholds_on_val(val_targets, val_probs)
    val_metrics_opt = calculate_metrics(val_targets, val_probs, threshold=opt_thresholds)

    print(f"\n🎯 Fold 9 Calibrated Thresholds: {dict(zip(TARGET_CLASSES, [round(float(x), 2) for x in opt_thresholds]))}")
    print(f"  >> Val Fold 9 (@0.50): Macro-F1 = {val_metrics_05['macro_f1']*100:.2f}% | Macro-AUC = {val_metrics_05['macro_auc']:.4f}")
    print(f"  >> Val Fold 9 (@Calib): Macro-F1 = {val_metrics_opt['macro_f1']*100:.2f}% | Macro-Sens = {val_metrics_opt['macro_sensitivity']*100:.2f}% | Macro-Spec = {val_metrics_opt['macro_specificity']*100:.2f}%")

    test_metrics_05, test_probs, test_targets = evaluate(model, test_loader, device, thresholds=0.5)
    test_metrics_opt = calculate_metrics(test_targets, test_probs, threshold=opt_thresholds)

    print("\n🔬 Strict Zero-Leakage Test Set (Fold 10) Results:")
    print(f"  >> [@0.50 fixed]  Test Macro-F1: {test_metrics_05['macro_f1']*100:.2f}% | Macro-AUC: {test_metrics_05['macro_auc']:.4f} | Sens: {test_metrics_05['macro_sensitivity']*100:.2f}% | Spec: {test_metrics_05['macro_specificity']*100:.2f}%")
    print(f"  >> [@Val-Calib]   Test Macro-F1: {test_metrics_opt['macro_f1']*100:.2f}% | Macro-AUC: {test_metrics_opt['macro_auc']:.4f} | Sens: {test_metrics_opt['macro_sensitivity']*100:.2f}% | Spec: {test_metrics_opt['macro_specificity']*100:.2f}%")

    print("\n📊 Test Fold 10 Per-Territory Breakdown (@Val-Calibrated Thresholds):")
    for cls_name, m in test_metrics_opt["per_class"].items():
        print(f"  - {cls_name:6s} (th={m['threshold']:.2f}): F1 = {m['f1']*100:5.2f}% | AUC = {m['auc']:.4f} | Sens = {m['sensitivity']*100:5.2f}% | Spec = {m['specificity']*100:5.2f}% (N={m['support']})")

    # 6. Sparsification & Budget Audit
    if args.prune_epoch == 0:
        pruned_edges = model.bottleneck_kan.prune_edges(threshold=1e-3)
    else:
        pruned_edges = int((model.bottleneck_kan.edge_mask < 0.5).sum().item())
    total_edges = 64 * 24
    final_budget = model.count_parameters()
    print(f"\n✂️ Sparsity & TinyML Audit: {pruned_edges}/{total_edges} edges pruned ({pruned_edges/total_edges*100:.1f}%) | Active Params: {final_budget['active_total_params']:,} (<12K: {final_budget['within_budget']})")

    # Update best checkpoint to also carry calibrated thresholds
    checkpoint["val_thresholds"] = opt_thresholds.tolist()
    checkpoint["final_budget"] = final_budget
    torch.save(checkpoint, os.path.join(args.save_dir, "model_best.pth"))

    # 7. Save Metrics Summary
    output_summary = {
        "arch": args.arch,
        "alpha": alpha_val,
        "gamma": args.gamma,
        "best_epoch": best_epoch,
        "val_thresholds": opt_thresholds.tolist(),
        "val_metrics_05": val_metrics_05,
        "val_metrics_calibrated": val_metrics_opt,
        "test_metrics_05": test_metrics_05,
        "test_metrics_calibrated": test_metrics_opt,
        "budget": final_budget,
        "pruned_edges_ratio": pruned_edges / total_edges,
        "training_time_seconds": total_time,
        "history": history
    }
    with open(os.path.join(args.save_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)

    print(f"\n📁 Saved deliverables to: {args.save_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
