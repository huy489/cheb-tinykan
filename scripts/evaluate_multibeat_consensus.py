"""
Multi-Beat Record Consensus Evaluation & Training Comparison (PRD-01 Upgrade).
Tests whether aggregating multiple R-peak beats from each 10s ECG recording closes
the AC-01 / AC-02 gap WITHOUT increasing MCU parameter budget (<12K) or Peak SRAM (4.22 KB).

Protocol (Strict Zero-Leakage):
  - Aggregation rule & per-class thresholds calibrated ONLY on Validation Fold 9.
  - Frozen Test Fold 10 evaluated ONCE with selected Fold-9 configuration.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.model import ChebTinyKAN
from models.symbolic_distillation import SymbolicMorphometricDistiller
from data.ptbxl_dataset import MultiBeatCachedPTBXLDataset
from data.scp_mapping import TARGET_CLASSES
from train import calculate_metrics, optimize_thresholds_on_val


def infer_multibeat_logits(model, dataset: MultiBeatCachedPTBXLDataset, batch_size: int = 128) -> tuple:
    """
    Runs ChebTinyKAN across all valid beats in `dataset` and returns:
      raw_logits: (N, K, 5)
      n_valid: (N,)
      labels: (N, 5)
    """
    model.eval()
    N, K = dataset.beats_i16.shape[0], dataset.beats_i16.shape[1]
    all_logits = np.zeros((N, K, 5), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            chunk = dataset.beats_i16[start:end].astype(np.float32) / 32767.0  # (B, K, 12, 180)
            b_size = end - start
            flat = torch.from_numpy(chunk.reshape(b_size * K, 12, 180))
            logits_flat = model(flat).cpu().numpy().reshape(b_size, K, 5)
            all_logits[start:end] = logits_flat

    return all_logits, dataset.n_valid, dataset.labels, dataset.sqi_valid


def aggregate_record_probabilities(logits_bk: np.ndarray, n_valid: np.ndarray, sqi_valid: np.ndarray, method: str) -> np.ndarray:
    """
    Aggregates K beat predictions per record into (N, 5) probabilities.
    Zero-SQI records output sigmoid of zero beat cleanly (EDGE-01 preserved).
    """
    N, K, C = logits_bk.shape
    probs_bk = 1.0 / (1.0 + np.exp(-np.clip(logits_bk, -30.0, 30.0)))
    out = np.zeros((N, C), dtype=np.float32)

    for i in range(N):
        if not sqi_valid[i] or n_valid[i] == 0:
            # Invalid SQI record -> beat 0 is all zeros per EDGE-01
            out[i] = probs_bk[i, 0]
            continue

        nv = int(n_valid[i])
        l_sub = logits_bk[i, :nv]
        p_sub = probs_bk[i, :nv]

        if method == "mean_prob":
            out[i] = np.mean(p_sub, axis=0)
        elif method == "median_prob":
            out[i] = np.median(p_sub, axis=0)
        elif method == "mean_logit":
            mean_l = np.mean(l_sub, axis=0)
            out[i] = 1.0 / (1.0 + np.exp(-np.clip(mean_l, -30.0, 30.0)))
        elif method == "trimmed_mean_prob":
            if nv >= 4:
                sorted_p = np.sort(p_sub, axis=0)
                out[i] = np.mean(sorted_p[1:-1], axis=0)
            else:
                out[i] = np.mean(p_sub, axis=0)
        elif method == "top2_mean_prob":
            if nv >= 2:
                sorted_p = np.sort(p_sub, axis=0)
                out[i] = np.mean(sorted_p[-2:], axis=0)
            else:
                out[i] = p_sub[0]
        else:
            raise ValueError(f"Unknown method: {method}")

    return out


def evaluate_checkpoint_multibeat(ckpt_path: str, mb_cache_path: str, out_json: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", ckpt.get("config", {}).get("arch", "baseline"))
    model = ChebTinyKAN(in_channels=12, num_classes=5, degree=4, arch=arch)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    budget = model.count_parameters()

    print("=" * 72)
    print(f"🫀 MULTI-BEAT RECORD CONSENSUS EVALUATION: {ckpt_path}")
    print(f"Arch: {arch} | Active Params: {budget['active_total_params']:,} (<12K: {budget['within_budget']})")
    print("=" * 72)

    val_ds = MultiBeatCachedPTBXLDataset(mb_cache_path, folds=[9], mode="all_beats")
    test_ds = MultiBeatCachedPTBXLDataset(mb_cache_path, folds=[10], mode="all_beats")

    val_logits, val_nv, val_labels, val_sqi = infer_multibeat_logits(model, val_ds)
    test_logits, test_nv, test_labels, test_sqi = infer_multibeat_logits(model, test_ds)

    methods = ["mean_prob", "median_prob", "mean_logit", "trimmed_mean_prob", "top2_mean_prob"]
    val_candidates = {}
    best_method = None
    best_val_f1_05 = -1.0

    print("\n🔎 Selecting Aggregation Strategy Strictly on Validation Fold 9:")
    for m in methods:
        val_probs = aggregate_record_probabilities(val_logits, val_nv, val_sqi, m)
        m_05 = calculate_metrics(val_labels, val_probs, threshold=0.5)
        val_candidates[m] = {
            "val_macro_f1_05": m_05["macro_f1"],
            "val_macro_auc": m_05["macro_auc"],
            "val_macro_sens_05": m_05["macro_sensitivity"],
            "val_macro_spec_05": m_05["macro_specificity"],
        }
        print(f"  - {m:18s}: Val AUC = {m_05['macro_auc']:.4f} | Val F1@0.5 = {m_05['macro_f1']*100:.2f}% | Sens = {m_05['macro_sensitivity']*100:.2f}%")
        if m_05["macro_f1"] > best_val_f1_05:
            best_val_f1_05 = m_05["macro_f1"]
            best_method = m

    # Optimize per-class thresholds strictly on Val Fold 9 for winning aggregation rule
    val_probs_best = aggregate_record_probabilities(val_logits, val_nv, val_sqi, best_method)
    val_metrics_05 = calculate_metrics(val_labels, val_probs_best, threshold=0.5)
    opt_thresh = optimize_thresholds_on_val(val_labels, val_probs_best)
    val_metrics_cal = calculate_metrics(val_labels, val_probs_best, threshold=opt_thresh)

    # Evaluate Frozen Test Fold 10 ONCE
    test_probs = aggregate_record_probabilities(test_logits, test_nv, test_sqi, best_method)
    test_metrics_05 = calculate_metrics(test_labels, test_probs, threshold=0.5)
    test_metrics_cal = calculate_metrics(test_labels, test_probs, threshold=opt_thresh)

    # Audit EDGE-05 clinical triage distribution on Fold 10
    edge05_counts = {"CRITICAL_MI_ALERT": 0, "SUSPECTED_ISCHEMIA_MONITOR_TREND": 0, "NORMAL_SINUS_MONITORING": 0}
    for row_p in test_probs:
        st = SymbolicMorphometricDistiller.classify_clinical_state(row_p)["clinical_state"]
        edge05_counts[st] += 1

    print("\n" + "-" * 72)
    print(f"🏆 Selected Fold-9 Strategy: {best_method}")
    print(f"🎯 Fold 9 Thresholds: {dict(zip(TARGET_CLASSES, [round(float(x), 2) for x in opt_thresh]))}")
    print(f"📊 Fold 10 [@0.50 fixed]: Macro-AUC = {test_metrics_05['macro_auc']:.4f} | Macro-F1 = {test_metrics_05['macro_f1']*100:.2f}% | Sens = {test_metrics_05['macro_sensitivity']*100:.2f}% | Spec = {test_metrics_05['macro_specificity']*100:.2f}%")
    print(f"📊 Fold 10 [@Val-Calib] : Macro-AUC = {test_metrics_cal['macro_auc']:.4f} | Macro-F1 = {test_metrics_cal['macro_f1']*100:.2f}% | Sens = {test_metrics_cal['macro_sensitivity']*100:.2f}% | Spec = {test_metrics_cal['macro_specificity']*100:.2f}%")
    print("\n  Per-Territory Breakdown (@Val-Calib):")
    for cls_name, cm in test_metrics_cal["per_class"].items():
        print(f"    - {cls_name:5s} (th={cm['threshold']:.2f}): AUC={cm['auc']:.4f} | F1={cm['f1']*100:5.2f}% | Sens={cm['sensitivity']*100:5.2f}% | Spec={cm['specificity']*100:5.2f}%")
    print(f"\n🛡️  EDGE-05 Triage Breakdown (Fold 10 N={len(test_probs)}): {edge05_counts}")

    report = {
        "checkpoint": os.path.relpath(ckpt_path, REPO_ROOT),
        "multibeat_cache": os.path.relpath(mb_cache_path, REPO_ROOT),
        "arch": arch,
        "budget": budget,
        "val_strategy_sweep": val_candidates,
        "selected_method": best_method,
        "val_thresholds": [float(x) for x in opt_thresh],
        "val_metrics_05": val_metrics_05,
        "val_metrics_calibrated": val_metrics_cal,
        "test_metrics_05": test_metrics_05,
        "test_metrics_calibrated": test_metrics_cal,
        "edge05_fold10_distribution": edge05_counts,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"✅ Saved Multi-Beat Consensus Report to: {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Beat 10s Consensus Evaluation")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--mb_cache", type=str, default=os.path.join(REPO_ROOT, "data", "cache", "ptbxl_multibeat_250hz.npz"))
    parser.add_argument("--out", type=str, default=os.path.join(REPO_ROOT, "docs", "multibeat_consensus_report.json"))
    args = parser.parse_args()
    evaluate_checkpoint_multibeat(args.ckpt, args.mb_cache, args.out)
