"""
Local DSP Preprocessing & Cache Generator for PTB-XL (All 10 Folds).
Runs deterministic DSP preprocessing ONCE on local multi-core CPU:
  1. Verifies strict inter-patient zero-leakage across Train (Folds 1-8), Val (Fold 9), Test (Fold 10).
  2. Executes SWMA (W=151) + Zero-Phase Butterworth Bandpass (0.5-45Hz) + 50Hz Notch + SQI + Beat Slicer.
  3. Saves compressed numpy archive (~100 MB) to data/cache/ptbxl_preprocessed_250hz.npz.
  4. Exports SHA-256 integrity manifest to docs/cache_audit.json for reproducibility proof.
"""

import os
import sys
import csv
import json
import time
import hashlib
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

# Ensure repo root is on PYTHONPATH
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.scp_mapping import SCPMapper, TARGET_CLASSES
from data.dsp_preprocessing import EndToEndECGPreprocessor

# Global worker preprocessor initialized once per worker process
_WORKER_PREPROCESSOR = None


def _init_worker():
    global _WORKER_PREPROCESSOR
    _WORKER_PREPROCESSOR = EndToEndECGPreprocessor(target_fs=250.0, window_size=180)


def _process_single_record(task):
    idx, dat_path = task
    raw_signal = np.fromfile(dat_path, dtype=np.int16).reshape(-1, 12)
    beat, quality = _WORKER_PREPROCESSOR.process(raw_signal)
    return idx, beat.astype(np.float32), bool(quality["is_valid"])


def prepare_cache(data_dir: str, output_npz: str, audit_json: str, workers: int):
    t_start = time.time()
    csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ptbxl_database.csv not found at: {csv_path}")

    print("=" * 70)
    print("⚙️  CHEB-TINYKAN LOCAL DSP PREPROCESSOR & CACHE GENERATOR")
    print("=" * 70)
    print(f"Dataset root: {data_dir}")
    print(f"CPU workers:  {workers}")

    records = []
    train_patients, val_patients, test_patients = set(), set(), set()

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fold = int(row["strat_fold"])
                ecg_id = int(row["ecg_id"])
                patient_id = int(float(row["patient_id"]))
            except (ValueError, KeyError):
                continue

            filename_hr = row.get("filename_hr", "")
            dat_path = os.path.join(data_dir, filename_hr + ".dat")
            if not os.path.exists(dat_path):
                continue

            label_vec = SCPMapper.get_multilabel_vector(row.get("scp_codes", ""))
            records.append({
                "ecg_id": ecg_id,
                "patient_id": patient_id,
                "fold": fold,
                "label": label_vec,
                "dat_path": dat_path
            })

            if 1 <= fold <= 8:
                train_patients.add(patient_id)
            elif fold == 9:
                val_patients.add(patient_id)
            elif fold == 10:
                test_patients.add(patient_id)

    n_total = len(records)
    print(f"Total verified .dat records: {n_total}")

    # Strict Inter-patient Zero-Leakage Check (AC-03)
    overlap_tr_val = train_patients & val_patients
    overlap_tr_te = train_patients & test_patients
    overlap_val_te = val_patients & test_patients
    assert len(overlap_tr_val) == 0, f"LEAKAGE Train/Val: {len(overlap_tr_val)} patients!"
    assert len(overlap_tr_te) == 0, f"LEAKAGE Train/Test: {len(overlap_tr_te)} patients!"
    assert len(overlap_val_te) == 0, f"LEAKAGE Val/Test: {len(overlap_val_te)} patients!"
    print("✅ Inter-patient Zero-Leakage Verified: 0 shared patients across Train/Val/Test!")

    beats = np.zeros((n_total, 12, 180), dtype=np.float32)
    labels = np.array([r["label"] for r in records], dtype=np.float32)
    folds = np.array([r["fold"] for r in records], dtype=np.int32)
    ecg_ids = np.array([r["ecg_id"] for r in records], dtype=np.int32)
    patient_ids = np.array([r["patient_id"] for r in records], dtype=np.int32)
    sqi_valid = np.zeros((n_total,), dtype=bool)

    tasks = [(i, r["dat_path"]) for i, r in enumerate(records)]
    print(f"Running vectorized DSP across {n_total} records...")

    t_dsp = time.time()
    completed = 0
    with Pool(processes=workers, initializer=_init_worker) as pool:
        for idx, beat, is_valid in pool.imap_unordered(_process_single_record, tasks, chunksize=64):
            beats[idx] = beat
            sqi_valid[idx] = is_valid
            completed += 1
            if completed % 5000 == 0 or completed == n_total:
                print(f"  Processed {completed}/{n_total} ({completed/n_total*100:.1f}%) - {time.time() - t_dsp:.1f}s")

    dsp_time = time.time() - t_dsp
    os.makedirs(os.path.dirname(os.path.abspath(output_npz)), exist_ok=True)
    np.savez_compressed(
        output_npz,
        beats=beats,
        labels=labels,
        folds=folds,
        ecg_ids=ecg_ids,
        patient_ids=patient_ids,
        sqi_valid=sqi_valid
    )

    file_size_mb = os.path.getsize(output_npz) / (1024.0 * 1024.0)
    beats_sha256 = hashlib.sha256(beats.tobytes()).hexdigest()
    labels_sha256 = hashlib.sha256(labels.tobytes()).hexdigest()

    # Build scientific audit manifest
    def split_stats(fold_list):
        m = np.isin(folds, fold_list)
        cls_counts = {name: int(labels[m, ci].sum()) for ci, name in enumerate(TARGET_CLASSES)}
        return {
            "num_records": int(m.sum()),
            "num_patients": int(len(set(patient_ids[m].tolist()))),
            "sqi_valid_count": int(sqi_valid[m].sum()),
            "sqi_rejected_count": int((~sqi_valid[m]).sum()),
            "class_counts": cls_counts
        }

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dsp_time_seconds": round(dsp_time, 2),
        "total_time_seconds": round(time.time() - t_start, 2),
        "cache_file": os.path.relpath(output_npz, REPO_ROOT),
        "cache_size_mb": round(file_size_mb, 2),
        "tensor_shape": list(beats.shape),
        "dtype": "float32",
        "sha256_beats": beats_sha256,
        "sha256_labels": labels_sha256,
        "zero_patient_leakage_verified": True,
        "splits": {
            "train_folds_1_8": split_stats(list(range(1, 9))),
            "val_fold_9": split_stats([9]),
            "test_fold_10": split_stats([10])
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(audit_json)), exist_ok=True)
    with open(audit_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 70)
    print(f"🎉 Cache Ready: {output_npz} ({file_size_mb:.2f} MB)")
    print(f"🔐 SHA-256 (beats):  {beats_sha256}")
    print(f"🔐 SHA-256 (labels): {labels_sha256}")
    print(f"📋 Audit Manifest:   {audit_json}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute PTB-XL DSP cache locally")
    parser.add_argument("--data_dir", type=str, default=os.path.join(REPO_ROOT, "data", "ptb-xl"))
    parser.add_argument("--output", type=str, default=os.path.join(REPO_ROOT, "data", "cache", "ptbxl_preprocessed_250hz.npz"))
    parser.add_argument("--audit", type=str, default=os.path.join(REPO_ROOT, "docs", "cache_audit.json"))
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = parser.parse_args()

    prepare_cache(args.data_dir, args.output, args.audit, args.workers)
