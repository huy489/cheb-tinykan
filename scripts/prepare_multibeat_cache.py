"""
Local Multi-Beat DSP Preprocessing & Q1.15 Cache Generator for PTB-XL (All 10 Folds).
Extracts up to K=8 clean R-peak centered 180-sample beats per 10s ECG record with:
  - Inter-patient zero-leakage assertion across Train (Folds 1-8), Val (Fold 9), Test (Fold 10)
  - SWMA (W=151) + Zero-Phase Butterworth Bandpass (0.5-45Hz) + 50Hz Notch + SQI (EDGE-01/02)
  - EDGE-03 tachycardia Tukey right-tail tapering (HR > 150 bpm)
  - Compact int16 Q1.15 storage (-32767..+32767) at data/cache/ptbxl_multibeat_250hz.npz
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.scp_mapping import SCPMapper, TARGET_CLASSES
from data.dsp_preprocessing import EndToEndECGPreprocessor

_WORKER_PREPROCESSOR = None
_MAX_BEATS = 8


def _init_worker(max_beats: int):
    global _WORKER_PREPROCESSOR, _MAX_BEATS
    _MAX_BEATS = max_beats
    _WORKER_PREPROCESSOR = EndToEndECGPreprocessor(target_fs=250.0, window_size=180)


def _process_record_multibeat(task):
    idx, dat_path = task
    raw_signal = np.fromfile(dat_path, dtype=np.int16).reshape(-1, 12)
    beats_f32, n_valid, quality = _WORKER_PREPROCESSOR.process_multibeat(raw_signal, max_beats=_MAX_BEATS)
    # Quantize normalized [-1.0, 1.0] beat tensors cleanly to int16 Q1.15 for compact RAM footprint
    beats_i16 = np.clip(np.round(beats_f32 * 32767.0), -32767, 32767).astype(np.int16)
    hr_bpm = float(quality.get("hr_bpm", 75.0))
    return idx, beats_i16, int(n_valid), bool(quality["is_valid"]), hr_bpm


def prepare_multibeat_cache(data_dir: str, output_npz: str, audit_json: str, workers: int, max_beats: int = 8):
    t_start = time.time()
    csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ptbxl_database.csv not found at: {csv_path}")

    print("=" * 72)
    print("⚡ CHEB-TINYKAN LOCAL MULTI-BEAT DSP PREPROCESSOR (10s -> K BEATS/RECORD)")
    print("=" * 72)
    print(f"Dataset root: {data_dir} | Max beats/record: {max_beats} | Workers: {workers}")

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
                "dat_path": dat_path,
            })

            if 1 <= fold <= 8:
                train_patients.add(patient_id)
            elif fold == 9:
                val_patients.add(patient_id)
            elif fold == 10:
                test_patients.add(patient_id)

    assert len(train_patients & val_patients) == 0, "LEAKAGE Train/Val"
    assert len(train_patients & test_patients) == 0, "LEAKAGE Train/Test"
    assert len(val_patients & test_patients) == 0, "LEAKAGE Val/Test"
    print(f"✅ Zero-Leakage Verified across {len(records)} records (0 shared patients).")

    n_total = len(records)
    beats = np.zeros((n_total, max_beats, 12, 180), dtype=np.int16)
    n_valid_beats = np.zeros((n_total,), dtype=np.uint8)
    hr_bpms = np.zeros((n_total,), dtype=np.float32)
    labels = np.array([r["label"] for r in records], dtype=np.float32)
    folds = np.array([r["fold"] for r in records], dtype=np.int32)
    ecg_ids = np.array([r["ecg_id"] for r in records], dtype=np.int32)
    patient_ids = np.array([r["patient_id"] for r in records], dtype=np.int32)
    sqi_valid = np.zeros((n_total,), dtype=bool)

    tasks = [(i, r["dat_path"]) for i, r in enumerate(records)]
    t_dsp = time.time()
    completed = 0
    with Pool(processes=workers, initializer=_init_worker, initargs=(max_beats,)) as pool:
        for idx, b_i16, n_val, is_val, hr in pool.imap_unordered(_process_record_multibeat, tasks, chunksize=64):
            beats[idx] = b_i16
            n_valid_beats[idx] = n_val
            sqi_valid[idx] = is_val
            hr_bpms[idx] = hr
            completed += 1
            if completed % 5000 == 0 or completed == n_total:
                print(f"  Processed {completed}/{n_total} ({completed/n_total*100:.1f}%) - {time.time() - t_dsp:.1f}s")

    dsp_time = time.time() - t_dsp
    os.makedirs(os.path.dirname(os.path.abspath(output_npz)), exist_ok=True)
    np.savez_compressed(
        output_npz,
        beats=beats,
        n_valid_beats=n_valid_beats,
        hr_bpms=hr_bpms,
        labels=labels,
        folds=folds,
        ecg_ids=ecg_ids,
        patient_ids=patient_ids,
        sqi_valid=sqi_valid,
    )

    file_size_mb = os.path.getsize(output_npz) / (1024.0 * 1024.0)
    beats_sha256 = hashlib.sha256(beats.tobytes()).hexdigest()
    labels_sha256 = hashlib.sha256(labels.tobytes()).hexdigest()

    tachy_count = int(np.sum(hr_bpms > 150.0))
    mean_beats_valid = float(np.mean(n_valid_beats[sqi_valid]))

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dsp_time_seconds": round(dsp_time, 2),
        "total_time_seconds": round(time.time() - t_start, 2),
        "cache_file": os.path.relpath(output_npz, REPO_ROOT),
        "cache_size_mb": round(file_size_mb, 2),
        "tensor_shape": list(beats.shape),
        "dtype": "int16_q15",
        "max_beats_per_record": max_beats,
        "mean_valid_beats_per_record": round(mean_beats_valid, 2),
        "edge03_tachycardia_gt150bpm_records": tachy_count,
        "sha256_beats_int16": beats_sha256,
        "sha256_labels": labels_sha256,
        "zero_patient_leakage_verified": True,
        "total_extracted_beats": int(np.sum(n_valid_beats)),
        "class_names": TARGET_CLASSES,
    }

    os.makedirs(os.path.dirname(os.path.abspath(audit_json)), exist_ok=True)
    with open(audit_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 72)
    print(f"🎉 Multi-Beat Cache Ready: {output_npz} ({file_size_mb:.2f} MB)")
    print(f"📊 Mean beats/record: {mean_beats_valid:.2f} | Total clean beats: {int(np.sum(n_valid_beats)):,} | HR>150 (EDGE-03): {tachy_count}")
    print(f"📋 Audit Manifest: {audit_json}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute PTB-XL Multi-Beat Q1.15 cache locally")
    parser.add_argument("--data_dir", type=str, default=os.path.join(REPO_ROOT, "data", "ptb-xl"))
    parser.add_argument("--output", type=str, default=os.path.join(REPO_ROOT, "data", "cache", "ptbxl_multibeat_250hz.npz"))
    parser.add_argument("--audit", type=str, default=os.path.join(REPO_ROOT, "docs", "multibeat_cache_audit.json"))
    parser.add_argument("--max_beats", type=int, default=8)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = parser.parse_args()

    prepare_multibeat_cache(args.data_dir, args.output, args.audit, args.workers, args.max_beats)
