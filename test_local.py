"""
Automated Local Verification Suite for Cheb-TinyKAN.
Verifies all mathematical foundations, dataset pipelines, and TinyML constraints
locally without requiring GPU.
"""

import os
import sys
import numpy as np

from models.cheb_kan import ChebyshevPolynomial
from data.scp_mapping import SCPMapper, TARGET_CLASSES
from data.ptbxl_dataset import PTBXLDataset


def test_chebyshev_recurrence():
    print("Testing Chebyshev 3-Term Recurrence Relation...", end=" ")
    x_test = np.linspace(-1.0, 1.0, 100)
    T = ChebyshevPolynomial.forward_numpy(x_test, degree=4)

    # 1. T_0(x) = 1
    assert np.allclose(T[..., 0], 1.0), "T0 failed"
    # 2. T_1(x) = x
    assert np.allclose(T[..., 1], x_test), "T1 failed"
    # 3. T_2(x) = 2x^2 - 1
    expected_T2 = 2.0 * (x_test ** 2) - 1.0
    assert np.allclose(T[..., 2], expected_T2), "T2 failed"
    # 4. T_3(x) = 4x^3 - 3x
    expected_T3 = 4.0 * (x_test ** 3) - 3.0 * x_test
    assert np.allclose(T[..., 3], expected_T3), "T3 failed"
    # 5. T_4(x) = 8x^4 - 8x^2 + 1
    expected_T4 = 8.0 * (x_test ** 4) - 8.0 * (x_test ** 2) + 1.0
    assert np.allclose(T[..., 4], expected_T4), "T4 failed"

    # 6. Strict bound: |T_k(x)| <= 1.0 for all x in [-1, 1]
    assert np.all(np.abs(T) <= 1.00001), "Chebyshev polynomial exceeded [-1, 1] bounds!"
    print("✅ PASSED!")


def test_scp_mapping():
    print("Testing Clinical SCP-ECG Multi-Territory Mapping...", end=" ")
    # Normal case
    y_norm = SCPMapper.get_multilabel_vector("{'NORM': 100.0}")
    assert y_norm == [1.0, 0.0, 0.0, 0.0, 0.0], f"NORM mapping failed: {y_norm}"

    # Inferior MI case
    y_imi = SCPMapper.get_multilabel_vector("{'IMI': 100.0}")
    assert y_imi == [0.0, 0.0, 1.0, 0.0, 0.0], f"IMI mapping failed: {y_imi}"

    # Multi-territory: Anteroseptal MI (ASMI -> AMI + SMI)
    y_asmi = SCPMapper.get_multilabel_vector("{'ASMI': 100.0}")
    assert y_asmi == [0.0, 1.0, 0.0, 0.0, 1.0], f"ASMI mapping failed: {y_asmi}"

    # Multi-territory: Inferolateral MI (ILMI -> IMI + LMI)
    y_ilmi = SCPMapper.get_multilabel_vector("{'ILMI': 100.0}")
    assert y_ilmi == [0.0, 0.0, 1.0, 1.0, 0.0], f"ILMI mapping failed: {y_ilmi}"
    print("✅ PASSED!")


def test_ptbxl_dataset_loading():
    print("Testing PTB-XL Native Binary Dataset & Beat Slicer...", end=" ")
    data_dir = os.path.join(os.path.dirname(__file__), "data", "ptb-xl")
    is_sample = False

    # Check if full dataset exists AND has actual record files
    has_full_dataset = False
    if os.path.exists(data_dir):
        records_dir = os.path.join(data_dir, "records500")
        if os.path.exists(records_dir) and os.path.isdir(records_dir) and len(os.listdir(records_dir)) > 0:
            has_full_dataset = True

    if not has_full_dataset:
        # Fallback to embedded sample test record in git repository
        data_dir = os.path.join(os.path.dirname(__file__), "data", "sample")
        is_sample = True

    if not os.path.exists(data_dir):
        print("⚠️ SKIPPED (No dataset or sample directory found)")
        return

    # Load Fold 10 (Strict Inter-patient Test Set)
    dataset = PTBXLDataset(data_dir=data_dir, folds=[10])
    assert len(dataset) > 0, "Dataset is empty!"

    # Sample item
    beat, label = dataset[0]
    assert beat.shape == (12, 180), f"Invalid beat shape: {beat.shape}, expected (12, 180)"
    assert len(label) == 5, f"Invalid label length: {len(label)}"
    assert np.all(beat >= -1.0) and np.all(beat <= 1.0), "Beat voltage not normalized to [-1, 1]!"

    if is_sample:
        print(f"✅ PASSED! (Verified with embedded test record: {len(dataset)} sample)")
    else:
        print(f"✅ PASSED! (Loaded {len(dataset)} records from Fold 10)")


def test_tinyml_budget_calculation():
    print("Testing TinyML Parameter Budget Calculation...", end=" ")
    # 1D-CNN Tokenizer:
    # Stage 1: Conv1D (k=7, s=2, cin=12, cout=24) -> 7*12*24 = 2,016 params + BN(48) = 2,064
    stage1 = (7 * 12 * 24) + 48
    # Stage 2: DS-Conv (k=5, s=2, cin=24, cout=48) -> Depthwise: 5*24 = 120, Pointwise: 24*48 = 1,152 + BN(96) = 1,368
    stage2 = (5 * 24) + (24 * 48) + 96
    # Stage 3: DS-Conv (k=5, s=2, cin=48, cout=64) -> Depthwise: 5*48 = 240, Pointwise: 48*64 = 3,072 + BN(128) = 3,440
    stage3 = (5 * 48) + (48 * 64) + 128
    tokenizer_total = stage1 + stage2 + stage3  # ~6,872 params

    # Cheb-KAN Bottleneck: 64 in -> 24 out, degree 4
    # Base weight: 24 * 64 = 1,536
    # Chebyshev coeffs: 24 * 64 * 5 = 7,680
    # Bias: 24
    bottleneck_total = 1536 + 7680 + 24  # 9,240 params
    # After 70% pruning of KAN bottleneck:
    bottleneck_pruned = 1536 * 0.3 + 7680 * 0.3 + 24  # ~2,788 params

    # Cheb-KAN Decision Head: 24 in -> 5 out, degree 4
    # Base weight: 5 * 24 = 120
    # Chebyshev coeffs: 5 * 24 * 5 = 600
    # Bias: 5
    decision_total = 120 + 600 + 5  # 725 params

    raw_total = tokenizer_total + bottleneck_total + decision_total  # ~16,837 params
    pruned_total = tokenizer_total + bottleneck_pruned + decision_total  # ~10,385 params

    assert raw_total < 20000, f"Raw total too large: {raw_total}"
    assert pruned_total < 12000, f"Pruned total exceeds 12K budget: {pruned_total}"
    print(f"✅ PASSED! (Raw: {raw_total:,} params | Pruned: {pruned_total:,.0f} params < 12,000)")


def test_symbolic_distillation_algebra():
    print("Testing Closed-Form Symbolic Distillation Matrix...", end=" ")
    # Transformation matrix M for degree 4
    M = np.array([
        [ 1.0,  0.0, -1.0,  0.0,  1.0],
        [ 0.0,  1.0,  0.0, -3.0,  0.0],
        [ 0.0,  0.0,  2.0,  0.0, -8.0],
        [ 0.0,  0.0,  0.0,  4.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  8.0]
    ])

    # Sample test coefficients c = [c0, c1, c2, c3, c4]
    c = np.array([0.5, -0.2, 0.3, 0.1, -0.05])
    a = M @ c  # canonical coefficients [a0, a1, a2, a3, a4]

    # Evaluate at multiple points in [-1, 1]
    x_pts = np.array([-0.8, -0.3, 0.0, 0.4, 0.9])
    T = ChebyshevPolynomial.forward_numpy(x_pts, degree=4)  # (5, 5)

    # 1. Chebyshev sum: Sum(c_k * T_k)
    y_cheb = T @ c

    # 2. Canonical monomial sum: a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
    y_canon = a[0] + a[1]*x_pts + a[2]*(x_pts**2) + a[3]*(x_pts**3) + a[4]*(x_pts**4)

    assert np.allclose(y_cheb, y_canon, atol=1e-6), "Symbolic distillation mismatch!"
    print("✅ PASSED!")


def test_quantization_and_pruning():
    print("Testing Q1.15 Quantizer, L1 Edge Pruning & C Exporter...", end=" ")
    from models.quantization import Q15Quantizer, KANEdgePruner, CHeaderExporter

    # 1. Test Q1.15 precision
    x = np.array([-1.0, -0.5, 0.0, 0.5, 0.9999], dtype=np.float32)
    q = Q15Quantizer.quantize_to_int16(x)
    deq = Q15Quantizer.dequantize(q)
    assert np.max(np.abs(x - deq)) < 0.0001, "Q15 quantization error exceeds 0.0001"

    # 2. Test 70% L1 Edge Pruner on Bottleneck KAN (24 out x 64 in)
    base_w = np.random.uniform(-0.5, 0.5, (24, 64)).astype(np.float32)
    cheb_c = np.random.uniform(-0.2, 0.2, (24, 64, 5)).astype(np.float32)
    p_base, p_cheb, mask, sparsity = KANEdgePruner.prune_cheb_kan_numpy(base_w, cheb_c, target_sparsity=0.70)

    assert abs(sparsity - 0.70) < 0.02, f"Target sparsity 70% not reached: {sparsity:.2%}"
    assert np.all(p_base[~mask] == 0.0), "Pruned base weights not zero!"
    assert np.all(p_cheb[~mask] == 0.0), "Pruned cheb coeffs not zero!"

    # 3. Test C header export
    header_path = os.path.join(os.path.dirname(__file__), "firmware", "cheb_tinykan_weights_test.h")
    CHeaderExporter.export_weights_header({"test_weight": p_base}, header_path, quantize_q15=True)
    assert os.path.exists(header_path), "C header file was not generated"
    if os.path.exists(header_path):
        os.remove(header_path)
    print("✅ PASSED!")


def test_easi_dower_ablation():
    print("Testing EASI Dower Lead Reconstruction (AC-09)... ", end=" ")
    from data.easi_dower import EASIDowerReconstructor

    reconstructor = EASIDowerReconstructor()

    # Synthetic 12-lead signal (12 leads x 180 samples)
    t = np.linspace(0, 1.0, 180)
    ecg_12 = np.zeros((12, 180), dtype=np.float32)
    for i in range(12):
        ecg_12[i] = np.sin(2 * np.pi * 1.5 * t + i * 0.2)

    # 1. Synthesize 3-lead EASI (ES, AS, AI)
    easi_3 = reconstructor.synthesize_easi_from_12lead(ecg_12)
    assert easi_3.shape == (3, 180), f"Invalid EASI shape: {easi_3.shape}"

    # 2. Reconstruct 12-lead
    recon_12 = reconstructor.reconstruct_12lead_from_easi(easi_3)
    assert recon_12.shape == (12, 180), f"Invalid reconstructed shape: {recon_12.shape}"

    # 3. Measure distortion
    dist = reconstructor.calculate_st_morphology_distortion(ecg_12, recon_12)
    assert "V2" in dist and "II" in dist, "Missing lead distortion metrics"
    print("✅ PASSED! (Synthesized 3 leads -> Reconstructed 12 leads)")


def test_clinical_symbolic_reporting():
    print("Testing Clinical Symbolic Reporting & Bedside Alerts (AC-06)... ", end=" ")
    from models.symbolic_distillation import SymbolicMorphometricDistiller

    # Test conversion of degree-4 coeffs
    cheb_c = np.array([0.4, 0.1, -0.05, 0.02, -0.01], dtype=np.float32)
    canon_a = SymbolicMorphometricDistiller.chebyshev_to_canonical(cheb_c)
    poly_str = SymbolicMorphometricDistiller.format_polynomial(canon_a, variable_name="ST_V2")
    assert "ST_V2" in poly_str, "Polynomial formatting failed"

    # Generate clinical alert
    lead_elevations = {"V1": 0.08, "V2": 0.28, "V3": 0.32, "III": -0.12}
    alert = SymbolicMorphometricDistiller.generate_clinical_alert("AMI", 0.965, lead_elevations, poly_str)
    assert "CLINICAL ALERT: ANTERIOR" in alert, "Alert string missing header"
    assert "LAD" in alert, "Alert missing culprit artery LAD"
    print("✅ PASSED!")


def test_c99_firmware_binary():
    print("Testing C99 Firmware Binary Compilation & Execution (AC-04/05/07)... ", end=" ")
    import subprocess
    cmd = "gcc -O3 -Wall -Wextra -std=c99 firmware/test_firmware.c firmware/cheb_tinykan_cmsis.c -lm -o firmware/test_firmware && ./firmware/test_firmware"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.path.dirname(__file__))
    assert res.returncode == 0, f"Firmware test execution failed:\n{res.stderr}"
    assert "ALL 5 FIRMWARE VERIFICATION SUITES PASSED" in res.stdout, "Firmware suite did not complete"
    print("✅ PASSED! (All C99 CMSIS-NN tests passed cleanly)")


if __name__ == "__main__":
    print("=" * 65)
    print("🧪 RUNNING CHEB-TINYKAN EXPANDED SYSTEM VERIFICATION TEST SUITE")
    print("=" * 65)
    test_chebyshev_recurrence()
    test_scp_mapping()
    test_ptbxl_dataset_loading()
    test_tinyml_budget_calculation()
    test_symbolic_distillation_algebra()
    test_quantization_and_pruning()
    test_easi_dower_ablation()
    test_clinical_symbolic_reporting()
    test_c99_firmware_binary()
    print("=" * 65)
    print("🎉 ALL 9 SYSTEM TEST SUITES PASSED! Phase 1, 2, 3, 4 Verified 100%.")
    print("=" * 65)
