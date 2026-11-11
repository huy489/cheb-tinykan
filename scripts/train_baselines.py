"""
IEEE Baseline Comparison Trainer (MLP-Tiny, ResNet1D, InceptionTime-1D).
Consumes identical pre-cached Inter-patient splits, --auto_alpha balanced Focal Loss,
and Fold-9 threshold calibration as Cheb-TinyKAN Run #2 for strict Pareto comparison.

Supports `--benchmark_epoch` to time 1 CPU epoch first so heavy models can be
routed cleanly to Colab T4 GPU if CPU speed is prohibitive (Hybrid Local-First Policy).
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.baselines import build_baseline, BASELINE_REGISTRY
from losses.focal_loss import MultiLabelFocalLoss
from data.ptbxl_dataset import create_dataloaders
from data.scp_mapping import TARGET_CLASSES
from train import set_seed, calculate_metrics, optimize_thresholds_on_val, compute_balanced_alpha_from_dataset


def estimate_activation_footprint_bytes(model: nn.Module, input_shape=(1, 12, 180), dtype_bytes: int = 2) -> int:
    """Estimates Peak single-layer output buffer bytes via forward hooks."""
    peaks = [int(np.prod(input_shape) * dtype_bytes)]
    hooks = []

    def _hook(_mod, _inp, out):
        if isinstance(out, torch.Tensor):
            peaks.append(int(out.numel() * dtype_bytes))

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            hooks.append(m.register_forward_hook(_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(input_shape))
    if was_training:
        model.train()

    for h in hooks:
        h.remove()
    return max(peaks)


def train_baseline_model(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=2 if device.type == "cuda" else 0,
        cache_path=args.cache_path,
    )

    alpha_val = compute_balanced_alpha_from_dataset(train_loader.dataset)
    model = build_baseline(args.model, in_channels=12, num_classes=5).to(device)
    total_params = int(sum(p.numel() for p in model.parameters()))
    peak_act_int16_bytes = estimate_activation_footprint_bytes(model.cpu(), dtype_bytes=2)
    peak_act_fp32_bytes = estimate_activation_footprint_bytes(model.cpu(), dtype_bytes=4)
    model = model.to(device)

    print("=" * 72)
    print(f"📊 IEEE BASELINE BENCHMARK: {args.model.upper()}")
    print(f"Params: {total_params:,} | Peak Act (INT16): {peak_act_int16_bytes/1024:.2f} KB | (FP32): {peak_act_fp32_bytes/1024:.2f} KB | Device: {device}")
    print("=" * 72)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = MultiLabelFocalLoss(alpha=alpha_val, gamma=args.gamma).to(device)

    if args.benchmark_epoch:
        t0 = time.time()
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        ep_sec = time.time() - t0
        print(f"⏱️  1-Epoch Timing Benchmark ({args.model} on {device}): {ep_sec:.2f}s/epoch (~{ep_sec*args.epochs/60:.1f} min for {args.epochs} epochs)")
        return

    best_val_f1 = 0.0
    best_epoch = 0
    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        model.train()
        total_loss = 0.0
        n_samp = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n_samp += x.size(0)
        scheduler.step()

        # Evaluate Val Fold 9
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for x, y in val_loader:
                vp.append(model.predict_probabilities(x.to(device)).cpu().numpy())
                vt.append(y.numpy())
        vp = np.concatenate(vp, axis=0)
        vt = np.concatenate(vt, axis=0)
        vm = calculate_metrics(vt, vp, threshold=0.5)
        ep_s = time.time() - t_ep

        is_best = vm["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = vm["macro_f1"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, os.path.join(args.save_dir, "model_best.pth"))

        history.append({"epoch": epoch, "train_loss": total_loss / max(1, n_samp), "val_macro_f1": vm["macro_f1"], "val_macro_auc": vm["macro_auc"]})
        if epoch == 1 or epoch % 10 == 0 or is_best:
            mark = "🔥" if is_best else "  "
            print(f"Ep [{epoch:02d}/{args.epochs:02d}] Loss={total_loss/n_samp:.4f} | Val F1@0.5={vm['macro_f1']*100:.2f}% | Val AUC={vm['macro_auc']:.4f} | {ep_s:.1f}s {mark}")

    total_time = time.time() - t_start
    best_ckpt = torch.load(os.path.join(args.save_dir, "model_best.pth"), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Final Val Fold 9 threshold calibration -> Frozen Fold 10 evaluation
    vp, vt = [], []
    tp, tt = [], []
    with torch.no_grad():
        for x, y in val_loader:
            vp.append(model.predict_probabilities(x.to(device)).cpu().numpy())
            vt.append(y.numpy())
        for x, y in test_loader:
            tp.append(model.predict_probabilities(x.to(device)).cpu().numpy())
            tt.append(y.numpy())
    vp, vt = np.concatenate(vp, 0), np.concatenate(vt, 0)
    tp, tt = np.concatenate(tp, 0), np.concatenate(tt, 0)

    opt_th = optimize_thresholds_on_val(vt, vp)
    val_m05 = calculate_metrics(vt, vp, threshold=0.5)
    val_mcal = calculate_metrics(vt, vp, threshold=opt_th)
    test_m05 = calculate_metrics(tt, tp, threshold=0.5)
    test_mcal = calculate_metrics(tt, tp, threshold=opt_th)

    print("\n🔬 Fold 10 Frozen Test Results:")
    print(f"  >> [@0.50 fixed]: F1={test_m05['macro_f1']*100:.2f}% | AUC={test_m05['macro_auc']:.4f} | Sens={test_m05['macro_sensitivity']*100:.2f}% | Spec={test_m05['macro_specificity']*100:.2f}%")
    print(f"  >> [@Val-Calib] : F1={test_mcal['macro_f1']*100:.2f}% | AUC={test_mcal['macro_auc']:.4f} | Sens={test_mcal['macro_sensitivity']*100:.2f}% | Spec={test_mcal['macro_specificity']*100:.2f}%")

    summary = {
        "model": args.model,
        "total_params": total_params,
        "within_12k_budget": total_params < 12000,
        "peak_act_int16_kb": round(peak_act_int16_bytes / 1024.0, 2),
        "peak_act_fp32_kb": round(peak_act_fp32_bytes / 1024.0, 2),
        "alpha": alpha_val,
        "gamma": args.gamma,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "val_thresholds": opt_th.tolist(),
        "val_metrics_05": val_m05,
        "val_metrics_calibrated": val_mcal,
        "test_metrics_05": test_m05,
        "test_metrics_calibrated": test_mcal,
        "training_time_seconds": total_time,
        "history": history,
    }
    with open(os.path.join(args.save_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved baseline summary to {args.save_dir}/training_metrics.json ({total_time/60:.2f} mins)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IEEE Baseline Comparison Runner")
    parser.add_argument("--model", type=str, required=True, choices=list(BASELINE_REGISTRY.keys()))
    parser.add_argument("--data_dir", type=str, default=os.path.join(REPO_ROOT, "data", "ptb-xl"))
    parser.add_argument("--cache_path", type=str, default=os.path.join(REPO_ROOT, "data", "cache", "ptbxl_preprocessed_250hz.npz"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark_epoch", action="store_true")
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = os.path.join(REPO_ROOT, "checkpoints", "baselines", args.model)
    train_baseline_model(args)
