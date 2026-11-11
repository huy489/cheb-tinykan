"""
C99 Weight Table Exporter for STM32F411RE & ESP32-S3 Firmware Deployment.
Converts a trained & Fold-9 calibrated Cheb-TinyKAN checkpoint (e.g. Run #2)
into static Flash C99 const arrays matching `ChebCalibratedModelWeights`.
"""

import os
import sys
import argparse
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.model import ChebTinyKAN
from data.ptbxl_dataset import CachedPTBXLDataset
from scripts.evaluate_clinical_and_edge import fold_conv_bn, calc_frac_bits, quantize_arr_int16, quantize_bias_int32


def _fmt_array(name: str, ctype: str, arr: np.ndarray, per_line: int = 16) -> str:
    flat = arr.flatten().tolist()
    lines = [f"static const {ctype} {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), per_line):
        chunk = ", ".join(str(int(v)) for v in flat[i:i + per_line])
        comma = "," if (i + per_line) < len(flat) else ""
        lines.append(f"    {chunk}{comma}")
    lines.append("};")
    return "\n".join(lines)


def export_c_weights(ckpt_path: str, cache_path: str, out_header: str, symbol: str = "g_cheb_calibrated_weights"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", ckpt.get("config", {}).get("arch", "baseline"))
    if arch != "baseline":
        raise ValueError("Firmware weight header exporter currently exports 'baseline' GAP architecture.")

    model = ChebTinyKAN(in_channels=12, num_classes=5, degree=4, arch=arch)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    budget = model.count_parameters()

    val_ds = CachedPTBXLDataset(cache_path, folds=[9])
    cal_x = torch.from_numpy(val_ds.beats[:256])

    with torch.no_grad():
        act1 = model.tokenizer.stage1(cal_x)
        act2_dw = model.tokenizer.stage2.depthwise(act1)
        act2_pw = model.tokenizer.stage2.act(model.tokenizer.stage2.bn(model.tokenizer.stage2.pointwise(act2_dw)))
        act3_dw = model.tokenizer.stage3.depthwise(act2_pw)
        act3_pw = model.tokenizer.stage3.act(model.tokenizer.stage3.bn(model.tokenizer.stage3.pointwise(act3_dw)))
        z_cal = model.tokenizer.gap(act3_pw).squeeze(-1)
        h_cal = model.bottleneck_kan(z_cal)
        logits_cal = model.decision_kan(h_cal)

    w_c1, b_c1 = fold_conv_bn(model.tokenizer.stage1[0], model.tokenizer.stage1[1])
    w_ds1_dw = model.tokenizer.stage2.depthwise.weight.detach().cpu().numpy()
    w_ds1_pw, b_ds1_pw = fold_conv_bn(model.tokenizer.stage2.pointwise, model.tokenizer.stage2.bn)
    w_ds2_dw = model.tokenizer.stage3.depthwise.weight.detach().cpu().numpy()
    w_ds2_pw, b_ds2_pw = fold_conv_bn(model.tokenizer.stage3.pointwise, model.tokenizer.stage3.bn)

    c1_bw = (model.bottleneck_kan.base_weight * model.bottleneck_kan.edge_mask).detach().cpu().numpy()
    c1_cc = (model.bottleneck_kan.cheb_coeffs * model.bottleneck_kan.edge_mask.unsqueeze(-1)).detach().cpu().numpy()
    c1_b = model.bottleneck_kan.bias.detach().cpu().numpy()
    c1_mask = (model.bottleneck_kan.edge_mask > 0.5).detach().cpu().numpy().astype(np.uint8)

    c2_bw = (model.decision_kan.base_weight * model.decision_kan.edge_mask).detach().cpu().numpy()
    c2_cc = (model.decision_kan.cheb_coeffs * model.decision_kan.edge_mask.unsqueeze(-1)).detach().cpu().numpy()
    c2_b = model.decision_kan.bias.detach().cpu().numpy()
    c2_mask = (model.decision_kan.edge_mask > 0.5).detach().cpu().numpy().astype(np.uint8)

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
    f_k1_w = calc_frac_bits(float(max(np.max(np.abs(c1_bw)), np.max(np.abs(c1_cc)))), max_frac=13)
    f_k1_o = calc_frac_bits(float(h_cal.abs().max()), max_frac=14)
    f_k2_w = calc_frac_bits(float(max(np.max(np.abs(c2_bw)), np.max(np.abs(c2_cc)))), max_frac=13)
    f_k2_o = calc_frac_bits(float(logits_cal.abs().max()), max_frac=13, min_frac=6)

    arrays = [
        ("s_c1_w", "int16_t", quantize_arr_int16(w_c1, f_c1_w)),
        ("s_c1_b", "int32_t", quantize_bias_int32(b_c1, 15 + f_c1_w)),
        ("s_ds1_dw_w", "int16_t", quantize_arr_int16(w_ds1_dw, f_ds1_dw_w)),
        ("s_ds1_pw_w", "int16_t", quantize_arr_int16(w_ds1_pw, f_ds1_pw_w)),
        ("s_ds1_pw_b", "int32_t", quantize_bias_int32(b_ds1_pw, f_ds1_dw_o + f_ds1_pw_w)),
        ("s_ds2_dw_w", "int16_t", quantize_arr_int16(w_ds2_dw, f_ds2_dw_w)),
        ("s_ds2_pw_w", "int16_t", quantize_arr_int16(w_ds2_pw, f_ds2_pw_w)),
        ("s_ds2_pw_b", "int32_t", quantize_bias_int32(b_ds2_pw, f_ds2_dw_o + f_ds2_pw_w)),
        ("s_k1_bw", "int16_t", quantize_arr_int16(c1_bw, f_k1_w)),
        ("s_k1_cc", "int16_t", quantize_arr_int16(c1_cc, f_k1_w)),
        ("s_k1_b", "int32_t", quantize_bias_int32(c1_b, 15 + f_k1_w)),
        ("s_k1_mask", "uint8_t", c1_mask),
        ("s_k2_bw", "int16_t", quantize_arr_int16(c2_bw, f_k2_w)),
        ("s_k2_cc", "int16_t", quantize_arr_int16(c2_cc, f_k2_w)),
        ("s_k2_b", "int32_t", quantize_bias_int32(c2_b, 15 + f_k2_w)),
        ("s_k2_mask", "uint8_t", c2_mask),
    ]

    val_th = ckpt.get("val_thresholds", [0.5, 0.5, 0.5, 0.5, 0.5])
    total_rodata_bytes = sum(a[2].nbytes for a in arrays)
    guard_name = os.path.basename(out_header).upper().replace(".", "_").replace("-", "_")

    parts = [
        "/* Auto-generated by scripts/export_c_weights.py — DO NOT EDIT MANUALLY */",
        f"/* Source checkpoint: {os.path.relpath(ckpt_path, REPO_ROOT)} | Active params: {budget['active_total_params']} */",
        f"/* Total Flash Q1.15/Q31 const table bytes (unpacked): {total_rodata_bytes} bytes */",
        f"#ifndef {guard_name}",
        f"#define {guard_name}",
        "",
        '#include "cheb_tinykan_cmsis.h"',
        "",
        f"static const float {symbol}_val_thresholds[5] = {{{', '.join(f'{float(x):.4f}f' for x in val_th)}}};",
        "",
    ]
    for name, ctype, arr in arrays:
        parts.append(_fmt_array(name, ctype, arr))
        parts.append("")

    parts.extend([
        "static const int16_t s_dummy_proj_w[1] = {0};",
        "static const int32_t s_dummy_proj_b[1] = {0};",
        "",
        f"static const ChebCalibratedModelWeights {symbol} = {{",
        "    .arch_mode = 0,",
        f"    .conv1_w = s_c1_w, .conv1_b = s_c1_b, .conv1_w_frac = {f_c1_w}, .conv1_out_frac = {f_c1_o},",
        f"    .ds1_dw_w = s_ds1_dw_w, .ds1_dw_w_frac = {f_ds1_dw_w}, .ds1_dw_out_frac = {f_ds1_dw_o},",
        f"    .ds1_pw_w = s_ds1_pw_w, .ds1_pw_b = s_ds1_pw_b, .ds1_pw_w_frac = {f_ds1_pw_w}, .ds1_pw_out_frac = {f_ds1_pw_o},",
        f"    .ds2_dw_w = s_ds2_dw_w, .ds2_dw_w_frac = {f_ds2_dw_w}, .ds2_dw_out_frac = {f_ds2_dw_o},",
        f"    .ds2_pw_w = s_ds2_pw_w, .ds2_pw_b = s_ds2_pw_b, .ds2_pw_w_frac = {f_ds2_pw_w}, .ds2_pw_out_frac = {f_ds2_pw_o},",
        "    .proj_w = s_dummy_proj_w, .proj_b = s_dummy_proj_b, .proj_w_frac = 12, .proj_out_frac = 12,",
        f"    .cheb1_base_w = s_k1_bw, .cheb1_coeffs = s_k1_cc, .cheb1_b = s_k1_b, .cheb1_edge_mask = s_k1_mask, .cheb1_w_frac = {f_k1_w}, .cheb1_out_frac = {f_k1_o},",
        f"    .cheb2_base_w = s_k2_bw, .cheb2_coeffs = s_k2_cc, .cheb2_b = s_k2_b, .cheb2_edge_mask = s_k2_mask, .cheb2_w_frac = {f_k2_w}, .cheb2_out_frac = {f_k2_o}",
        "};",
        "",
        f"#endif /* {guard_name} */",
        "",
    ])

    os.makedirs(os.path.dirname(os.path.abspath(out_header)), exist_ok=True)
    with open(out_header, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"✅ Exported calibrated Q-format weights ({os.path.relpath(ckpt_path, REPO_ROOT)}) -> {out_header} ({total_rodata_bytes} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export calibrated Cheb-TinyKAN C99 weights header")
    parser.add_argument("--ckpt", type=str, default=os.path.join(REPO_ROOT, "checkpoints", "run2_balanced_pruned", "model_best.pth"))
    parser.add_argument("--cache_path", type=str, default=os.path.join(REPO_ROOT, "data", "cache", "ptbxl_preprocessed_250hz.npz"))
    parser.add_argument("--out", type=str, default=os.path.join(REPO_ROOT, "firmware", "cheb_tinykan_weights_run2.h"))
    parser.add_argument("--symbol", type=str, default="g_cheb_run2_weights")
    args = parser.parse_args()
    export_c_weights(args.ckpt, args.cache_path, args.out, args.symbol)
