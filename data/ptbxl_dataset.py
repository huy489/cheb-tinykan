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


def create_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 2
) -> Tuple:
    """
    Creates standard Inter-patient Train, Validation, and Test loaders.
    - Train: Folds 1 - 8
    - Validation: Fold 9
    - Test: Fold 10 (Strict Inter-patient protocol)
    """
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch is required to create DataLoader instances.")

    train_dataset = PTBXLDataset(data_dir, folds=list(range(1, 9)))
    val_dataset = PTBXLDataset(data_dir, folds=[9])
    test_dataset = PTBXLDataset(data_dir, folds=[10])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
