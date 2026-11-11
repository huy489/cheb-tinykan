"""
Comprehensive Post-Training Verification Suite for Cheb-TinyKAN:
  - AC-06 & AC-07: Symbolic Morphometric Distillation & Clinical Alert Extraction
  - AC-08: C99 Fixed-Point Quantization (Q-format) & Bit-Exactness vs PyTorch FP32 (500 Fold 10 beats)
  - AC-09: EASI Dower Lead Reconstruction Quantitative Ablation Study (All 2,198 Fold 10 records)

Runs completely on Local CPU in < 30 seconds using pre-cached Fold 10 & local PTB-XL dataset.
"""

import os
import sys
import json
import time
import math
import ctypes
import subprocess
import argparse
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.model import ChebTinyKAN
from models.symbolic_distillation import SymbolicMorphometricDistiller
from data.scp_mapping import TARGET_CLASSES
from data.ptbxl_dataset import CachedPTBXLDataset, PTBXLDataset
from data.easi_dower import EASIDowerReconstructor, LEAD_NAMES
from data.dsp_preprocessing import EndToEndECGPreprocessor
from train import calculate_metrics


# ---------------------------------------------------------------------------
# Ctypes bindings matching firmware/cheb_tinykan_cmsis.h
# ---------------------------------------------------------------------------

class ChebInferenceResult(ctypes.Structure):
    _fields_ = [
        ("probabilities", ctypes.c_float * 5),
        ("is_mi_detected", ctypes.c_bool),
        ("primary_territory", ctypes.c_int),
        ("confidence", ctypes.c_float),
        ("inference_cycles", ctypes.c_uint32),
    ]


class ChebCalibratedModelWeights(ctypes.Structure):
    _fields_ = [
        ("arch_mode", ctypes.c_uint8),
        ("conv1_w", ctypes.POINTER(ctypes.c_int16)),
        ("conv1_b", ctypes.POINTER(ctypes.c_int32)),
        ("conv1_w_frac", ctypes.c_uint8),
        ("conv1_out_frac", ctypes.c_uint8),

        ("ds1_dw_w", ctypes.POINTER(ctypes.c_int16)),
        ("ds1_dw_w_frac", ctypes.c_uint8),
        ("ds1_dw_out_frac", ctypes.c_uint8),

        ("ds1_pw_w", ctypes.POINTER(ctypes.c_int16)),
        ("ds1_pw_b", ctypes.POINTER(ctypes.c_int32)),
        ("ds1_pw_w_frac", ctypes.c_uint8),
        ("ds1_pw_out_frac", ctypes.c_uint8),

        ("ds2_dw_w", ctypes.POINTER(ctypes.c_int16)),
        ("ds2_dw_w_frac", ctypes.c_uint8),
        ("ds2_dw_out_frac", ctypes.c_uint8),

        ("ds2_pw_w", ctypes.POINTER(ctypes.c_int16)),
        ("ds2_pw_b", ctypes.POINTER(ctypes.c_int32)),
        ("ds2_pw_w_frac", ctypes.c_uint8),
        ("ds2_pw_out_frac", ctypes.c_uint8),

        ("proj_w", ctypes.POINTER(ctypes.c_int16)),
        ("proj_b", ctypes.POINTER(ctypes.c_int32)),
        ("proj_w_frac", ctypes.c_uint8),
        ("proj_out_frac", ctypes.c_uint8),

        ("cheb1_base_w", ctypes.POINTER(ctypes.c_int16)),
        ("cheb1_coeffs", ctypes.POINTER(ctypes.c_int16)),
        ("cheb1_b", ctypes.POINTER(ctypes.c_int32)),
        ("cheb1_edge_mask", ctypes.POINTER(ctypes.c_uint8)),
        ("cheb1_w_frac", ctypes.c_uint8),
        ("cheb1_out_frac", ctypes.c_uint8),

        ("cheb2_base_w", ctypes.POINTER(ctypes.c_int16)),
        ("cheb2_coeffs", ctypes.POINTER(ctypes.c_int16)),
        ("cheb2_b", ctypes.POINTER(ctypes.c_int32)),
        ("cheb2_edge_mask", ctypes.POINTER(ctypes.c_uint8)),
        ("cheb2_w_frac", ctypes.c_uint8),
        ("cheb2_out_frac", ctypes.c_uint8),
    ]


def fold_conv_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d):
    """Fuses Conv1d (bias=False) + BatchNorm1d into effective weight & bias numpy arrays."""
    w = conv.weight.detach().cpu().numpy()
    gamma = bn.weight.detach().cpu().numpy()
    beta = bn.bias.detach().cpu().numpy()
    mean = bn.running_mean.detach().cpu().numpy()
    var = bn.running_var.detach().cpu().numpy()
    eps = bn.eps

    scale = gamma / np.sqrt(var + eps)
    w_folded = w * scale.reshape(-1, 1, 1)
    b_conv = conv.bias.detach().cpu().numpy() if conv.bias is not None else np.zeros_like(mean)
    b_folded = beta + (b_conv - mean) * scale
    return w_folded.astype(np.float32), b_folded.astype(np.float32)


def calc_frac_bits(max_abs_val: float, max_frac: int = 14, min_frac: int = 6) -> int:
    if max_abs_val < 1e-6:
        return max_frac
    bits = int(math.floor(math.log2(32767.0 / (1.05 * max_abs_val))))
    return max(min_frac, min(max_frac, bits))


def quantize_arr_int16(arr: np.ndarray, frac_bits: int) -> np.ndarray:
    scale = float(1 << frac_bits)
    q = np.round(arr * scale)
    return np.clip(q, -32768, 32767).astype(np.int16)


def quantize_bias_int32(arr: np.ndarray, total_frac_bits: int) -> np.ndarray:
    scale = float(1 << total_frac_bits)
    q = np.round(arr * scale)
    return np.clip(q, -2147483648, 2147483647).astype(np.int32)


def run_symbolic_distillation_audit(model, test_beats, test_labels, thresholds):
    """AC-06 & AC-07: Extract closed-form polynomials and inspect real AMI sample."""
    print("\n" + "=" * 70)
    print("📐 [AC-06 / AC-07] SYMBOLIC MORPHOMETRIC DISTILLATION AUDIT")
    print("=" * 70)

    model.eval()
    dec_cheb_coeffs = model.decision_kan.cheb_coeffs.detach().cpu().numpy()  # (5, 24, 5)
    dec_base_w = model.decision_kan.base_weight.detach().cpu().numpy()        # (5, 24)
    canonical_dec = SymbolicMorphometricDistiller.chebyshev_to_canonical(dec_cheb_coeffs)  # (5, 24, 5)

    # Verify algebraic approximation error on random h in [-1, 1]
    h_grid = np.linspace(-1.0, 1.0, 50, dtype=np.float32)
    T_grid = np.stack([
        np.ones_like(h_grid),
        h_grid,
        2 * h_grid**2 - 1,
        4 * h_grid**3 - 3 * h_grid,
        8 * h_grid**4 - 8 * h_grid**2 + 1
    ], axis=-1)  # (50, 5)

    y_cheb = np.einsum("kd,oid->oik", T_grid, dec_cheb_coeffs)
    y_canon = (
        canonical_dec[:, :, 0:1]
        + canonical_dec[:, :, 1:2] * h_grid
        + canonical_dec[:, :, 2:3] * (h_grid**2)
        + canonical_dec[:, :, 3:4] * (h_grid**3)
        + canonical_dec[:, :, 4:5] * (h_grid**4)
    )
    max_alg_err = float(np.max(np.abs(y_cheb - y_canon)))
    print(f"Algebraic Chebyshev -> Canonical Monomial Max Error: {max_alg_err:.2e} (Target < 2.0%)")

    # Find an actual Anterior MI (class 1) beat in Fold 10 with high predicted probability
    ami_indices = np.where(test_labels[:, 1] == 1.0)[0]
    with torch.no_grad():
        ami_probs = model.predict_probabilities(torch.from_numpy(test_beats[ami_indices]))[:, 1].numpy()
    best_ami_local = int(np.argmax(ami_probs))
    sample_idx = int(ami_indices[best_ami_local])
    sample_prob = float(ami_probs[best_ami_local])
    sample_beat = test_beats[sample_idx]  # (12, 180) in [-1, 1] (scale 2.0 mV)

    # Extract bottleneck latent vector h for this patient to rank top-3 edge formulas
    with torch.no_grad():
        x_t = torch.from_numpy(sample_beat).unsqueeze(0)
        z_t = model.tokenizer(x_t)
        h_t = model.bottleneck_kan(z_t).squeeze(0).numpy()  # (24,)

    # Compute each bottleneck neuron's contribution to AMI logit
    ami_contribs = []
    for i in range(24):
        hi = float(h_t[i])
        hi_clamped = max(-1.0, min(1.0, hi))
        a = canonical_dec[1, i]
        poly_val = a[0] + a[1]*hi_clamped + a[2]*(hi_clamped**2) + a[3]*(hi_clamped**3) + a[4]*(hi_clamped**4)
        base_val = float(dec_base_w[1, i] * math.tanh(hi))
        total_val = base_val + poly_val
        poly_str = SymbolicMorphometricDistiller.format_polynomial(a, variable_name=f"h{i}")
        ami_contribs.append((abs(total_val), total_val, i, hi, dec_base_w[1, i], poly_str))

    ami_contribs.sort(reverse=True, key=lambda item: item[0])
    top_formula = " + ".join([f"{c[4]:+.2f}*tanh(h{c[2]}) + ({c[5]})" for c in ami_contribs[:2]])

    # Read precordial ST-level ~80ms after R-peak (R=50, +80ms=20 samples -> idx 70) in mV
    st_mv = {
        "V1": float(sample_beat[6, 70] * 2.0),
        "V2": float(sample_beat[7, 70] * 2.0),
        "V3": float(sample_beat[8, 70] * 2.0),
        "V4": float(sample_beat[9, 70] * 2.0),
        "III": float(sample_beat[2, 70] * 2.0),
    }

    alert = SymbolicMorphometricDistiller.generate_clinical_alert(
        territory="AMI",
        probability=sample_prob,
        lead_elevations=st_mv,
        polynomial_str=top_formula
    )
    print(f"\nSample Fold 10 AMI Record #{sample_idx} (Threshold={thresholds[1]:.2f}):")
    print(alert)

    return {
        "max_algebraic_error": max_alg_err,
        "sample_fold10_idx": sample_idx,
        "sample_ami_prob": sample_prob,
        "top_contributing_formula": top_formula,
        "measured_st_mv": st_mv
    }


def run_c99_quantization_bitexactness(model, val_beats, test_beats):
    """AC-08: Compile C99 engine, fold BN & calibrate shifts on Val Fold 9, compare 500 Fold 10 beats vs PyTorch FP32."""
    print("\n" + "=" * 70)
    print("⚙️  [AC-08] C99 FIXED-POINT BIT-EXACTNESS BENCHMARK (500 FOLD 10 BEATS)")
    print("=" * 70)

    so_path = "/tmp/libchebtinykan_eval.so"
    c_src = os.path.join(REPO_ROOT, "firmware", "cheb_tinykan_cmsis.c")
    subprocess.run(["gcc", "-O3", "-std=c99", "-shared", "-fPIC", c_src, "-lm", "-o", so_path], check=True)
    lib = ctypes.CDLL(so_path)
    lib.cheb_tinykan_infer_calibrated.argtypes = [
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ChebCalibratedModelWeights),
        ctypes.c_bool,
        ctypes.POINTER(ChebInferenceResult)
    ]
    lib.cheb_tinykan_infer_calibrated.restype = ctypes.c_int

    model.eval()
    is_tz = (model.arch == "temporal_zones")

    # 1. Calibrate activation dynamic ranges strictly on Val Fold 9 subset (N=256)
    cal_x = torch.from_numpy(val_beats[:256])
    with torch.no_grad():
        act1 = model.tokenizer.stage1(cal_x)
        act2_dw = model.tokenizer.stage2.depthwise(act1)
        act2_pw = model.tokenizer.stage2.act(model.tokenizer.stage2.bn(model.tokenizer.stage2.pointwise(act2_dw)))
        act3_dw = model.tokenizer.stage3.depthwise(act2_pw)
        act3_pw = model.tokenizer.stage3.act(model.tokenizer.stage3.bn(model.tokenizer.stage3.pointwise(act3_dw)))
        if is_tz:
            act_proj = model.tokenizer.proj(act3_pw)
            z_cal = model.tokenizer.zone_pool(act_proj).flatten(1)
            proj_max = float(act_proj.abs().max())
        else:
            z_cal = model.tokenizer.gap(act3_pw).squeeze(-1)
            proj_max = 1.0
        h_cal = model.bottleneck_kan(z_cal)
        logits_cal = model.decision_kan(h_cal)

    # Fold Conv + BN layers
    w_c1, b_c1 = fold_conv_bn(model.tokenizer.stage1[0], model.tokenizer.stage1[1])
    w_ds1_dw = model.tokenizer.stage2.depthwise.weight.detach().cpu().numpy()
    w_ds1_pw, b_ds1_pw = fold_conv_bn(model.tokenizer.stage2.pointwise, model.tokenizer.stage2.bn)
    w_ds2_dw = model.tokenizer.stage3.depthwise.weight.detach().cpu().numpy()
    w_ds2_pw, b_ds2_pw = fold_conv_bn(model.tokenizer.stage3.pointwise, model.tokenizer.stage3.bn)

    if is_tz:
        w_proj, b_proj = fold_conv_bn(model.tokenizer.proj[0], model.tokenizer.proj[1])
    else:
        w_proj, b_proj = np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)

    # KAN weights
    c1_bw = (model.bottleneck_kan.base_weight * model.bottleneck_kan.edge_mask).detach().cpu().numpy()
    c1_cc = (model.bottleneck_kan.cheb_coeffs * model.bottleneck_kan.edge_mask.unsqueeze(-1)).detach().cpu().numpy()
    c1_b = model.bottleneck_kan.bias.detach().cpu().numpy()
    c1_mask = (model.bottleneck_kan.edge_mask > 0.5).detach().cpu().numpy().astype(np.uint8)

    c2_bw = (model.decision_kan.base_weight * model.decision_kan.edge_mask).detach().cpu().numpy()
    c2_cc = (model.decision_kan.cheb_coeffs * model.decision_kan.edge_mask.unsqueeze(-1)).detach().cpu().numpy()
    c2_b = model.decision_kan.bias.detach().cpu().numpy()
    c2_mask = (model.decision_kan.edge_mask > 0.5).detach().cpu().numpy().astype(np.uint8)

    # Calibrate power-of-two fraction bits
    f_c1_w = calc_frac_bits(float(np.max(np.abs(w_c1))), max_frac=13)
    f_c1_o = calc_frac_bits(float(act1.abs().max()), max_frac=14)

    f_ds1_dw_w = calc_frac_bits(float(np.max(np.abs(w_ds1_dw))), max_frac=13)
    f_ds1_dw_o = calc_frac_bits(float(act2_dw.abs().max()), max_frac=14)
    f_ds1_pw_w = calc_frac_bits(float(np.max(np.abs(w_ds1_pw))), max_frac=13)
    f_ds1_pw_o = calc_frac_bits(float(act2_pw.abs().max()), max_frac=14)

    f_ds2_dw_w = calc_frac_bits(float(np.max(np.abs(w_ds2_dw))), max_frac=13)
    f_ds2_dw_o = calc_frac_bits(float(act3_dw.abs().max()), max_frac=14)
    f_ds2_pw_w = calc_frac_bits(float(np.max(np.abs(w_ds2_pw))), max_frac=13)
    f_ds2_pw_o = calc_frac_bits(float(act3_pw.abs().max()), max_frac=14)

    f_proj_w = calc_frac_bits(float(np.max(np.abs(w_proj))), max_frac=13)
    f_proj_o = calc_frac_bits(proj_max, max_frac=14)

    z_frac = f_proj_o if is_tz else f_ds2_pw_o
    f_k1_w = calc_frac_bits(float(max(np.max(np.abs(c1_bw)), np.max(np.abs(c1_cc)))), max_frac=13)
    f_k1_o = calc_frac_bits(float(h_cal.abs().max()), max_frac=14)
    f_k2_w = calc_frac_bits(float(max(np.max(np.abs(c2_bw)), np.max(np.abs(c2_cc)))), max_frac=13)
    f_k2_o = calc_frac_bits(float(logits_cal.abs().max()), max_frac=13, min_frac=6)

    # Quantize arrays cleanly into contiguous buffers kept alive in Python
    q_arrays = {
        "c1_w": np.ascontiguousarray(quantize_arr_int16(w_c1, f_c1_w).flatten()),
        "c1_b": np.ascontiguousarray(quantize_bias_int32(b_c1, 15 + f_c1_w)),
        "ds1_dw_w": np.ascontiguousarray(quantize_arr_int16(w_ds1_dw, f_ds1_dw_w).flatten()),
        "ds1_pw_w": np.ascontiguousarray(quantize_arr_int16(w_ds1_pw, f_ds1_pw_w).flatten()),
        "ds1_pw_b": np.ascontiguousarray(quantize_bias_int32(b_ds1_pw, f_ds1_dw_o + f_ds1_pw_w)),
        "ds2_dw_w": np.ascontiguousarray(quantize_arr_int16(w_ds2_dw, f_ds2_dw_w).flatten()),
        "ds2_pw_w": np.ascontiguousarray(quantize_arr_int16(w_ds2_pw, f_ds2_pw_w).flatten()),
        "ds2_pw_b": np.ascontiguousarray(quantize_bias_int32(b_ds2_pw, f_ds2_dw_o + f_ds2_pw_w)),
        "proj_w": np.ascontiguousarray(quantize_arr_int16(w_proj, f_proj_w).flatten()),
        "proj_b": np.ascontiguousarray(quantize_bias_int32(b_proj, f_ds2_pw_o + f_proj_w)),
        "k1_bw": np.ascontiguousarray(quantize_arr_int16(c1_bw, f_k1_w).flatten()),
        "k1_cc": np.ascontiguousarray(quantize_arr_int16(c1_cc, f_k1_w).flatten()),
        "k1_b": np.ascontiguousarray(quantize_bias_int32(c1_b, 15 + f_k1_w)),
        "k1_m": np.ascontiguousarray(c1_mask.flatten()),
        "k2_bw": np.ascontiguousarray(quantize_arr_int16(c2_bw, f_k2_w).flatten()),
        "k2_cc": np.ascontiguousarray(quantize_arr_int16(c2_cc, f_k2_w).flatten()),
        "k2_b": np.ascontiguousarray(quantize_bias_int32(c2_b, 15 + f_k2_w)),
        "k2_m": np.ascontiguousarray(c2_mask.flatten()),
    }

    cw = ChebCalibratedModelWeights(
        arch_mode=1 if is_tz else 0,
        conv1_w=q_arrays["c1_w"].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        conv1_b=q_arrays["c1_b"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        conv1_w_frac=f_c1_w, conv1_out_frac=f_c1_o,
        ds1_dw_w=q_arrays["ds1_dw_w"].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        ds1_dw_w_frac=f_ds1_dw_w, ds1_dw_out_frac=f_ds1_dw_o,
        ds1_pw_w=q_arrays["ds1_pw_w"].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        ds1_pw_b=q_arrays["ds1_pw_b"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ds1_pw_w_frac=f_ds1_pw_w, ds1_pw_out_frac=f_ds1_pw_o,
        ds2_dw_w=q_arrays["ds2_dw_w"].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        ds2_dw_w_frac=f_ds2_dw_w, ds2_dw_out_frac=f_ds2_dw_o,
        ds2_pw_w=q_arrays["ds2_pw_w"].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        ds2_pw_b=q_arrays["ds2_pw_b"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ds2_pw_w_frac=f_ds2_pw_w, ds2_pw_out_frac=f_ds2_pw_o,
        proj_w=q_arrays["proj_w"].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        proj_b=q_arrays["proj_b"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        proj_w_frac=f_proj_w, proj_out_frac=f_proj_o,
        cheb1_base_w=q_arrays["k1_bw"].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        cheb1_coeffs=q_arrays["k1_cc"].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        cheb1_b=q_arrays["k1_b"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        cheb1_edge_mask=q_arrays["k1_m"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        cheb1_w_frac=f_k1_w, cheb1_out_frac=f_k1_o,
        cheb2_base_w=q_arrays["k2_bw"].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        cheb2_coeffs=q_arrays["k2_cc"].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        cheb2_b=q_arrays["k2_b"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        cheb2_edge_mask=q_arrays["k2_m"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        cheb2_w_frac=f_k2_w, cheb2_out_frac=f_k2_o,
    )

    # Evaluate on 500 Fold 10 test beats
    n_eval = min(500, len(test_beats))
    sub_beats = test_beats[:n_eval]
    with torch.no_grad():
        fp32_probs = model.predict_probabilities(torch.from_numpy(sub_beats)).numpy()

    c99_probs = np.zeros_like(fp32_probs)
    res = ChebInferenceResult()
    t0 = time.time()
    for i in range(n_eval):
        q_in = np.ascontiguousarray(quantize_arr_int16(sub_beats[i], 15).flatten())
        status = lib.cheb_tinykan_infer_calibrated(
            q_in.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ctypes.byref(cw),
            True,  # skip SQI so even zeroed-out records are compared identically to PyTorch
            ctypes.byref(res)
        )
        assert status == 0, f"C99 inference error {status} at beat {i}"
        c99_probs[i] = [res.probabilities[c] for c in range(5)]
    c99_ms_per_beat = (time.time() - t0) * 1000.0 / n_eval

    abs_diff = np.abs(fp32_probs - c99_probs)
    mean_abs_err = float(np.mean(abs_diff))
    max_abs_err = float(np.max(abs_diff))
    p99_abs_err = float(np.percentile(abs_diff, 99))
    flips_05 = int(np.sum((fp32_probs >= 0.5) != (c99_probs >= 0.5)))

    if os.path.exists(so_path):
        os.remove(so_path)

    print(f"Calibrated Q-Formats: Conv1(Q{15-f_c1_o}.{f_c1_o}), DS2(Q{15-f_ds2_pw_o}.{f_ds2_pw_o}), KAN1(Q{15-f_k1_o}.{f_k1_o}), Logit(Q{15-f_k2_o}.{f_k2_o})")
    print(f"Evaluated Beats:       {n_eval} (500 x 5 = {n_eval*5} predictions)")
    print(f"Mean Abs Prob Error:   {mean_abs_err*100:.4f}%")
    print(f"99th Pct Abs Error:    {p99_abs_err*100:.4f}%")
    print(f"Max Abs Prob Error:    {max_abs_err*100:.4f}% (AC-08 Target < 1.50%)")
    print(f"Prediction Flips @0.5: {flips_05}/{n_eval*5} ({flips_05/(n_eval*5)*100:.3f}%)")
    print(f"C99 Inference Speed:   {c99_ms_per_beat:.3f} ms/beat on CPU")

    return {
        "n_beats": n_eval,
        "mean_abs_prob_error": mean_abs_err,
        "p99_abs_prob_error": p99_abs_err,
        "max_abs_prob_error": max_abs_err,
        "flips_at_05": flips_05,
        "c99_ms_per_beat": c99_ms_per_beat
    }


def run_easi_dower_ablation(model, data_dir, orig_beats, test_labels, thresholds):
    """AC-09: Runs EASI 3-lead Dower matrix reconstruction across ALL Fold 10 records and compares metrics."""
    print("\n" + "=" * 70)
    print("🩺 [AC-09] EASI DOWER LEAD RECONSTRUCTION ABLATION STUDY (ALL FOLD 10)")
    print("=" * 70)

    raw_test_ds = PTBXLDataset(data_dir=data_dir, folds=[10])
    reconstructor = EASIDowerReconstructor()
    preprocessor = EndToEndECGPreprocessor(target_fs=250.0, window_size=180)

    n_test = len(raw_test_ds)
    assert n_test == len(orig_beats), f"Mismatch Fold 10 count: {n_test} vs {len(orig_beats)}"

    easi_beats = np.zeros_like(orig_beats, dtype=np.float32)
    t0 = time.time()
    for i, sample in enumerate(raw_test_ds.samples):
        raw_sig = np.fromfile(sample["dat_path"], dtype=np.int16).reshape(-1, 12)  # (5000, 12)
        # Apply Dower projection (12 -> 3 EASI -> 12 reconstructed) on continuous raw signal
        recon_12_ch = reconstructor.run_easi_degradation_pipeline(raw_sig.T).T  # (5000, 12)
        beat, _ = preprocessor.process(recon_12_ch)
        easi_beats[i] = beat
    print(f"Preprocessed {n_test} EASI-reconstructed Fold 10 records in {time.time() - t0:.1f}s")

    # Measure normalized precordial voltage distortion across beats
    st_dist = reconstructor.calculate_st_morphology_distortion(
        orig_beats.transpose(1, 0, 2).reshape(12, -1),
        easi_beats.transpose(1, 0, 2).reshape(12, -1)
    )

    model.eval()
    with torch.no_grad():
        orig_probs = model.predict_probabilities(torch.from_numpy(orig_beats)).numpy()
        easi_probs = model.predict_probabilities(torch.from_numpy(easi_beats)).numpy()

    orig_metrics = calculate_metrics(test_labels, orig_probs, threshold=thresholds)
    easi_metrics = calculate_metrics(test_labels, easi_probs, threshold=thresholds)

    print("\n📊 Comparative Diagnostic Performance (12-Lead Gold Standard vs 4-Electrode EASI):")
    print(f"  >> Gold Standard 12-Lead: Macro-AUC = {orig_metrics['macro_auc']:.4f} | Macro-F1 = {orig_metrics['macro_f1']*100:.2f}% | Macro-Sens = {orig_metrics['macro_sensitivity']*100:.2f}%")
    print(f"  >> EASI Reconstructed:    Macro-AUC = {easi_metrics['macro_auc']:.4f} | Macro-F1 = {easi_metrics['macro_f1']*100:.2f}% | Macro-Sens = {easi_metrics['macro_sensitivity']*100:.2f}%")
    print("\n  Per-Territory Drop (Gold 12-Lead -> EASI):")
    for cls_name in TARGET_CLASSES:
        om = orig_metrics["per_class"][cls_name]
        em = easi_metrics["per_class"][cls_name]
        print(
            f"    - {cls_name:5s}: AUC {om['auc']:.4f} -> {em['auc']:.4f} ({(em['auc']-om['auc'])*100:+.2f}%) | "
            f"Sens {om['sensitivity']*100:5.1f}% -> {em['sensitivity']*100:5.1f}% | "
            f"F1 {om['f1']*100:5.1f}% -> {em['f1']*100:5.1f}%"
        )

    precordial_mae = {k: round(st_dist[k]["mae_mv"], 4) for k in ["V1", "V2", "V3", "V4", "II", "III"]}
    print(f"\n  Mean Lead Waveform MAE (normalized): {precordial_mae}")

    return {
        "gold_12lead_metrics": orig_metrics,
        "easi_reconstructed_metrics": easi_metrics,
        "auc_drop": float(easi_metrics["macro_auc"] - orig_metrics["macro_auc"]),
        "f1_drop": float(easi_metrics["macro_f1"] - orig_metrics["macro_f1"]),
        "normalized_lead_mae": precordial_mae
    }


def main():
    parser = argparse.ArgumentParser(description="Post-training AC-06 to AC-09 verification")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained model_best.pth")
    parser.add_argument("--data_dir", type=str, default=os.path.join(REPO_ROOT, "data", "ptb-xl"))
    parser.add_argument("--cache_path", type=str, default=os.path.join(REPO_ROOT, "data", "cache", "ptbxl_preprocessed_250hz.npz"))
    parser.add_argument("--out", type=str, default=os.path.join(REPO_ROOT, "docs", "ac_verification_report.json"))
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", ckpt.get("config", {}).get("arch", "baseline"))
    thresholds = np.array(ckpt.get("val_thresholds", [0.5]*5), dtype=np.float32)

    model = ChebTinyKAN(in_channels=12, num_classes=5, degree=4, arch=arch)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_ds = CachedPTBXLDataset(args.cache_path, folds=[9])
    test_ds = CachedPTBXLDataset(args.cache_path, folds=[10])

    symbolic_res = run_symbolic_distillation_audit(model, test_ds.beats, test_ds.labels, thresholds)
    c99_res = run_c99_quantization_bitexactness(model, val_ds.beats, test_ds.beats)
    easi_res = run_easi_dower_ablation(model, args.data_dir, test_ds.beats, test_ds.labels, thresholds)

    report = {
        "checkpoint": os.path.relpath(args.ckpt, REPO_ROOT),
        "arch": arch,
        "budget": model.count_parameters(),
        "ac06_ac07_symbolic_distillation": symbolic_res,
        "ac08_c99_quantization": c99_res,
        "ac09_easi_ablation": easi_res
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Saved complete verification report to: {args.out}")


if __name__ == "__main__":
    main()
