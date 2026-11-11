"""
PTB-XL Dataset & DataLoader Module for Cheb-TinyKAN.
Supports strict Inter-patient Split (Folds 1-8 Train, Fold 9 Val, Fold 10 Test).
Performs:
- Native fast binary int16 parsing without external heavy dependencies.
- Downsampling 500Hz -> 250Hz.
- Beat segmentation (180 samples: -200ms to +520ms around R-peak).
- Voltage normalization to [-1.0, 1.0] for Chebyshev polynomial domain.
"""

import os
import csv
import numpy as np
from typing import List, Tuple, Optional, Union
from .scp_mapping import SCPMapper
from .dsp_preprocessing import EndToEndECGPreprocessor

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    BASE_DATASET = Dataset
except ImportError:
    torch = None
    DataLoader = None
    BASE_DATASET = object


class PTBXLDataset(BASE_DATASET):
    """
    PyTorch Dataset for PTB-XL ECG records.
    """

    def __init__(
        self,
        data_dir: str,
        folds: List[int],
        window_size: int = 180,
        downsample_factor: int = 2,
        voltage_scale: float = 2.0
    ):
        """
        Args:
            data_dir: Path to directory containing ptbxl_database.csv and records500/
            folds: List of fold integers (1 to 10) to include
            window_size: Number of samples per beat window (default 180 at 250Hz)
            downsample_factor: Downsample from 500Hz to 250Hz by taking every 2nd sample
            voltage_scale: Full scale voltage (in mV) mapped to range [-1.0, 1.0]
        """
        self.data_dir = data_dir
        self.folds = set(folds)
        self.window_size = window_size
        self.downsample_factor = downsample_factor
        self.voltage_scale = voltage_scale
        self.preprocessor = EndToEndECGPreprocessor(target_fs=250.0, window_size=window_size)

        self.samples = []
        self._load_metadata()

    def _load_metadata(self):
        csv_path = os.path.join(self.data_dir, "ptbxl_database.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"ptbxl_database.csv not found at: {csv_path}")

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    fold = int(row["strat_fold"])
                except (ValueError, KeyError):
                    continue

                if fold in self.folds:
                    filename_hr = row.get("filename_hr", "")
                    scp_codes = row.get("scp_codes", "")
                    label_vec = SCPMapper.get_multilabel_vector(scp_codes)

                    dat_path = os.path.join(self.data_dir, filename_hr + ".dat")
                    if os.path.exists(dat_path):
                        self.samples.append({
                            "ecg_id": row.get("ecg_id", ""),
                            "patient_id": row.get("patient_id", ""),
                            "dat_path": dat_path,
                            "label": label_vec,
                            "fold": fold
                        })

    def __len__(self) -> int:
        return len(self.samples)

    def extract_beat(self, ecg_raw: np.ndarray) -> np.ndarray:
        """
        Executes complete DSP Preprocessing Pipeline:
        SWMA Baseline Wander Filter -> Butterworth Bandpass (0.5-45Hz) + 50Hz Notch -> SQI -> Beat Slicer -> Normalize
        """
        beat, quality = self.preprocessor.process(ecg_raw)
        return beat


    def __getitem__(self, idx: int):
        item = self.samples[idx]
        raw_signal = np.fromfile(item["dat_path"], dtype=np.int16).reshape(-1, 12)
        beat = self.extract_beat(raw_signal)
        label = np.array(item["label"], dtype=np.float32)

        if torch is not None:
            return torch.from_numpy(beat), torch.from_numpy(label)
        return beat, label


class CachedPTBXLDataset(BASE_DATASET):
    """
    High-speed RAM-cached Dataset backed by precomputed DSP outputs (.npz).
    Eliminates redundant per-epoch CPU signal processing while keeping strict fold isolation.
    """

    def __init__(self, cache_path: str, folds: List[int]):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Preprocessed cache not found at: {cache_path}")

        with np.load(cache_path) as data:
            mask = np.isin(data["folds"], list(folds))
            self.beats = np.ascontiguousarray(data["beats"][mask], dtype=np.float32)
            self.labels = np.ascontiguousarray(data["labels"][mask], dtype=np.float32)
            self.folds_arr = data["folds"][mask]

    def __len__(self) -> int:
        return len(self.beats)

    def __getitem__(self, idx: int):
        beat = self.beats[idx]
        label = self.labels[idx]
        if torch is not None:
            return torch.from_numpy(beat), torch.from_numpy(label)
        return beat, label


class MultiBeatCachedPTBXLDataset(BASE_DATASET):
    """
    RAM-efficient int16 Q1.15 Multi-Beat Dataset backed by ptbxl_multibeat_250hz.npz.
    mode='random_beat': samples 1 random valid beat from the record's K detected beats (Train augmentation).
    mode='all_beats': returns (K, 12, 180) float32 tensor, n_valid count, and label (Val/Test consensus).
    """

    def __init__(self, cache_path: str, folds: List[int], mode: str = "random_beat", seed: int = 42):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Multi-beat cache not found at: {cache_path}")
        if mode not in ("random_beat", "all_beats"):
            raise ValueError(f"Unsupported mode={mode}")

        with np.load(cache_path) as data:
            mask = np.isin(data["folds"], list(folds))
            self.beats_i16 = np.ascontiguousarray(data["beats"][mask], dtype=np.int16)
            self.n_valid = np.ascontiguousarray(data["n_valid_beats"][mask], dtype=np.int64)
            self.labels = np.ascontiguousarray(data["labels"][mask], dtype=np.float32)
            self.sqi_valid = np.ascontiguousarray(data["sqi_valid"][mask], dtype=bool)
            self.hr_bpms = np.ascontiguousarray(data["hr_bpms"][mask], dtype=np.float32)

        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.beats_i16)

    def __getitem__(self, idx: int):
        label = self.labels[idx]
        nv = int(self.n_valid[idx])
        if self.mode == "random_beat":
            k = int(self.rng.integers(0, nv)) if nv > 1 else 0
            beat = self.beats_i16[idx, k].astype(np.float32) / 32767.0
            if torch is not None:
                return torch.from_numpy(beat), torch.from_numpy(label)
            return beat, label

        beats_all = self.beats_i16[idx].astype(np.float32) / 32767.0
        if torch is not None:
            return torch.from_numpy(beats_all), nv, torch.from_numpy(label)
        return beats_all, nv, label


def create_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 2,
    cache_path: Optional[str] = None
) -> Tuple:
    """
    Creates standard Inter-patient Train, Validation, and Test loaders.
    - Train: Folds 1 - 8
    - Validation: Fold 9
    - Test: Fold 10 (Strict Inter-patient protocol)

    Automatically uses precomputed RAM cache if `cache_path` exists (or default data/cache/*.npz).
    """
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch is required to create DataLoader instances.")

    # Auto-detect default cache file next to data_dir if not explicitly overridden
    if cache_path is None:
        default_cache = os.path.join(os.path.dirname(os.path.abspath(data_dir)), "cache", "ptbxl_preprocessed_250hz.npz")
        if os.path.exists(default_cache):
            cache_path = default_cache

    use_pin_memory = torch.cuda.is_available()

    if cache_path is not None and os.path.exists(cache_path):
        print(f"⚡ Loading preprocessed RAM cache from: {cache_path}")
        train_dataset = CachedPTBXLDataset(cache_path, folds=list(range(1, 9)))
        val_dataset = CachedPTBXLDataset(cache_path, folds=[9])
        test_dataset = CachedPTBXLDataset(cache_path, folds=[10])
        effective_workers = 0  # Zero worker overhead since data is already contiguous in RAM
    else:
        train_dataset = PTBXLDataset(data_dir, folds=list(range(1, 9)))
        val_dataset = PTBXLDataset(data_dir, folds=[9])
        test_dataset = PTBXLDataset(data_dir, folds=[10])
        effective_workers = num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=effective_workers,
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=use_pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=use_pin_memory
    )

    return train_loader, val_loader, test_loader
