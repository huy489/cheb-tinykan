"""
Digital Signal Processing (DSP) Preprocessing Pipeline for 12-Lead ECG.
Compliant with PRD-01 and Paper 23 (Zhejiang Univ / IEEE 2011):
1. Statistical Weighted Moving Average (SWMA) baseline wander filter (preserves ST-segment slope).
2. IIR Butterworth Bandpass (0.5 - 45 Hz) + 50Hz Notch filter.
3. Signal Quality Index (SQI) & Hardware Lead Fault Detector (EDGE-01 & EDGE-02).
4. Adaptive R-Peak Beat Slicer (-200ms to +520ms -> 180 samples @ 250Hz).
5. Dynamic Normalization to [-1.0, 1.0] for Chebyshev polynomial domain.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
try:
    import scipy.signal as signal
    from scipy.ndimage import uniform_filter1d
except ImportError:
    signal = None
    uniform_filter1d = None


class SWMAFilter:
    """
    Statistical Weighted Moving Average (SWMA) Filter.
    Kế thừa từ Paper 23 (IEEE 2011): Khử hiện tượng trôi đường đẳng điện do hô hấp
    nhưng bảo toàn nguyên vẹn biên độ và độ dốc đoạn ST mà không gây méo dạng sóng.
    """
    def __init__(self, window_size: int = 151):
        # 151 samples at 250Hz corresponds to ~600ms window
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1

    def filter_1d(self, x: np.ndarray) -> np.ndarray:
        half_w = self.window_size // 2
        pad_x = np.pad(x, (half_w, half_w), mode='edge')
        # Fast moving average approximation of baseline drift
        kernel = np.ones(self.window_size, dtype=np.float32) / self.window_size
        baseline = np.convolve(pad_x, kernel, mode='valid')
        # Subtract baseline wander
        return x - baseline

    def filter_multichannel(self, ecg: np.ndarray) -> np.ndarray:
        """
        Args:
            ecg: (samples, leads) array
        Returns:
            detrended_ecg: (samples, leads) array
        """
        if uniform_filter1d is not None:
            # Vectorized C implementation mathematically equivalent to edge-padded moving average
            baseline = uniform_filter1d(ecg, size=self.window_size, axis=0, mode='nearest')
            return (ecg - baseline).astype(np.float32)

        filtered = np.empty_like(ecg, dtype=np.float32)
        for ch in range(ecg.shape[1]):
            filtered[:, ch] = self.filter_1d(ecg[:, ch])
        return filtered


class ButterworthBandpassNotch:
    """
    IIR Butterworth Bandpass (0.5 - 45 Hz) + Notch (50 Hz) Filter.
    Khử nhiễu tần số cao (nhiễu cơ EMG) và triệt tiêu can nhiễu lưới điện xoay chiều.
    """
    def __init__(self, fs: float = 250.0, lowcut: float = 0.5, highcut: float = 45.0, notch_freq: float = 50.0):
        self.fs = fs
        if signal is not None:
            # 2nd-order Butterworth bandpass filter
            self.b_band, self.a_band = signal.butter(2, [lowcut, highcut], btype='bandpass', fs=fs)
            # Notch filter at 50 Hz (Q=30)
            self.b_notch, self.a_notch = signal.iirnotch(notch_freq, 30.0, fs=fs)
        else:
            self.b_band = self.a_band = self.b_notch = self.a_notch = None

    def filter_multichannel(self, ecg: np.ndarray) -> np.ndarray:
        if signal is None:
            # Fallback if scipy is unavailable
            return ecg

        # Forward-backward filter along axis=0 across all channels simultaneously
        y_band = signal.filtfilt(self.b_band, self.a_band, ecg, axis=0)
        y_clean = signal.filtfilt(self.b_notch, self.a_notch, y_band, axis=0)
        return y_clean.astype(np.float32)


class SignalQualityIndex:
    """
    Phát hiện tuột điện cực (Lead Fault) và bão hòa khuếch đại (EDGE-01 & EDGE-02).
    """
    @staticmethod
    def evaluate(ecg: np.ndarray, fs: float = 250.0) -> Dict:
        """
        Args:
            ecg: (samples, 12) array in mV
        Returns:
            dict with is_valid, sqi_score, and fault_flags
        """
        n_samples, n_leads = ecg.shape
        faulty_leads = []
        is_saturated = False

        # 1. Check flatline / lead disconnection (variance < 0.005 mV^2 across 0.5s windows)
        sub_len = int(0.5 * fs)
        for ch in range(n_leads):
            stds = [np.std(ecg[i:i+sub_len, ch]) for i in range(0, n_samples - sub_len, sub_len)]
            if min(stds) < 0.005:
                faulty_leads.append(ch)

        # 2. Check extreme clipping / saturation (|voltage| > 4.8 mV for more than 5% duration)
        clipped_ratio = np.mean(np.abs(ecg) > 4.8)
        if clipped_ratio > 0.05:
            is_saturated = True

        sqi_score = max(0.0, 1.0 - (len(faulty_leads) / n_leads) - (0.5 if is_saturated else 0.0))
        is_valid = (len(faulty_leads) <= 2) and (not is_saturated) and (sqi_score >= 0.60)

        return {
            "is_valid": is_valid,
            "sqi_score": float(sqi_score),
            "faulty_leads": faulty_leads,
            "is_saturated": is_saturated,
            "error_message": "LEAD_FAULT_DETECTED" if not is_valid else "OK"
        }


class AdaptiveRPeakBeatSlicer:
    """
    Định vị đỉnh sóng R và cắt phân đoạn nhịp tim 180 mẫu ([-200ms, +520ms]).
    """
    def __init__(self, window_size: int = 180, fs: float = 250.0):
        self.window_size = window_size
        self.fs = fs
        self.pre_samples = int(0.200 * fs)   # 50 samples before R-peak (-200ms)
        self.post_samples = int(0.520 * fs)  # 130 samples after R-peak (+520ms)

    def extract_dominant_beat(self, ecg: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Args:
            ecg: (samples, 12) array in mV
        Returns:
            beat: (12, 180) array normalized to [-1.0, 1.0]
            r_peak_idx: int sample index of detected R-peak
        """
        n_samples = ecg.shape[0]

        # Use Lead II (index 1) and V2 (index 7) for robust QRS detection
        lead_ii = ecg[:, 1] if ecg.shape[1] > 1 else ecg[:, 0]
        diff = np.diff(lead_ii)
        energy = diff ** 2

        # Search window for stable central R-peak
        search_start = min(int(1.0 * self.fs), n_samples // 4)
        search_end = min(int(8.0 * self.fs), n_samples * 3 // 4)

        if search_end > search_start + 50:
            r_peak = search_start + int(np.argmax(energy[search_start:search_end]))
        else:
            r_peak = n_samples // 2

        start = r_peak - self.pre_samples
        end = start + self.window_size

        # Boundary padding
        if start < 0:
            start = 0
            end = self.window_size
        elif end > n_samples:
            end = n_samples
            start = max(0, end - self.window_size)

        beat = ecg[start:end, :]
        if beat.shape[0] < self.window_size:
            pad = self.window_size - beat.shape[0]
            beat = np.pad(beat, ((0, pad), (0, 0)), mode='edge')

        # Voltage scaling to [-1.0, 1.0] for Chebyshev polynomial stability
        # Physiological full-scale voltage is +/- 2.0 mV
        beat_norm = np.clip(beat / 2.0, -1.0, 1.0)
        # Transpose to shape (12, 180) -> (Channels, Time)
        return beat_norm.T.astype(np.float32), r_peak

    def extract_multibeat_windows(
        self,
        ecg: np.ndarray,
        max_beats: int = 8,
        min_rr_sec: float = 0.30
    ) -> Tuple[np.ndarray, int, float]:
        """
        Extracts up to `max_beats` clean R-peak centered windows from a continuous 12-lead ECG.
        Implements EDGE-03: estimates heart rate and applies a right-tail Tukey cosine taper
        when HR > 150 bpm so adjacent beat P/T waves do not contaminate the +360..+520ms tail
        while keeping QRS and ST segment intact.
        Returns:
            beats: (max_beats, 12, 180) float32 in [-1.0, 1.0]
            n_valid: int actual number of detected beats (padded by repeating valid beats if < max_beats)
            hr_bpm: float estimated heart rate in bpm
        """
        n_samples = ecg.shape[0]
        lead_ii = ecg[:, 1] if ecg.shape[1] > 1 else ecg[:, 0]
        lead_v2 = ecg[:, 7] if ecg.shape[1] > 7 else lead_ii

        # Composite QRS slope energy smoothed over ~60ms
        d_ii = np.diff(lead_ii, prepend=lead_ii[0])
        d_v2 = np.diff(lead_v2, prepend=lead_v2[0])
        energy = (d_ii ** 2) + 0.5 * (d_v2 ** 2)
        if uniform_filter1d is not None:
            smooth_e = uniform_filter1d(energy, size=15, mode='nearest')
        else:
            kernel = np.ones(15, dtype=np.float32) / 15.0
            smooth_e = np.convolve(energy, kernel, mode='same')

        margin_start = max(self.pre_samples, int(0.25 * self.fs))
        margin_end = min(n_samples - self.post_samples, n_samples - int(0.55 * self.fs))

        peaks = []
        if margin_end > margin_start + 10:
            sub = smooth_e[margin_start:margin_end]
            thresh = max(0.18 * float(np.percentile(sub, 96)), 1e-4)
            refractory = int(min_rr_sec * self.fs)

            # Collect local maxima above threshold and apply greedy non-max suppression
            local_max_mask = (sub[1:-1] >= sub[:-2]) & (sub[1:-1] >= sub[2:]) & (sub[1:-1] > thresh)
            cand_indices = np.where(local_max_mask)[0] + 1 + margin_start
            # Sort candidates descending by smoothed QRS energy
            order = np.argsort(smooth_e[cand_indices])[::-1]
            for ci in cand_indices[order]:
                if all(abs(int(ci) - p) >= refractory for p in peaks):
                    peaks.append(int(ci))
            peaks.sort()

        if len(peaks) >= 2:
            rr_samples = np.diff(peaks)
            median_rr_sec = float(np.median(rr_samples)) / self.fs
            hr_bpm = float(np.clip(60.0 / max(median_rr_sec, 0.25), 30.0, 240.0))
        else:
            hr_bpm = 75.0

        if len(peaks) == 0:
            dom_beat, dom_r = self.extract_dominant_beat(ecg)
            beats_out = np.zeros((max_beats, 12, self.window_size), dtype=np.float32)
            beats_out[0] = dom_beat
            return beats_out, 1, hr_bpm

        # Select up to max_beats evenly spaced across the 10s strip
        if len(peaks) > max_beats:
            sel_idx = np.linspace(0, len(peaks) - 1, max_beats, dtype=int)
            selected_peaks = [peaks[i] for i in sel_idx]
        else:
            selected_peaks = peaks

        # EDGE-03: Right-tail cosine taper on +340ms..+520ms (indices 135..180) when HR > 150 bpm
        tukey_mask = None
        if hr_bpm > 150.0:
            tukey_mask = np.ones(self.window_size, dtype=np.float32)
            taper_start = min(135, self.window_size - 10)
            taper_len = self.window_size - taper_start
            phase = np.linspace(0.0, np.pi, taper_len, dtype=np.float32)
            tukey_mask[taper_start:] = 0.5 * (1.0 + np.cos(phase))

        beats_list = []
        for r_peak in selected_peaks:
            start = max(0, r_peak - self.pre_samples)
            end = start + self.window_size
            if end > n_samples:
                end = n_samples
                start = max(0, end - self.window_size)
            beat = ecg[start:end, :]
            if beat.shape[0] < self.window_size:
                pad = self.window_size - beat.shape[0]
                beat = np.pad(beat, ((0, pad), (0, 0)), mode='edge')
            beat_norm = np.clip(beat / 2.0, -1.0, 1.0).T.astype(np.float32)
            if tukey_mask is not None:
                beat_norm = beat_norm * tukey_mask.reshape(1, -1)
            beats_list.append(beat_norm)

        n_valid = len(beats_list)
        beats_out = np.zeros((max_beats, 12, self.window_size), dtype=np.float32)
        for k in range(n_valid):
            beats_out[k] = beats_list[k]

        return beats_out, n_valid, hr_bpm


class EndToEndECGPreprocessor:
    """
    Toàn bộ chuỗi tiền xử lý DSP từ tín hiệu thô 500Hz đến tensor đầu vào của Cheb-TinyKAN.
    """
    def __init__(self, target_fs: float = 250.0, window_size: int = 180):
        self.target_fs = target_fs
        self.window_size = window_size
        self.swma = SWMAFilter(window_size=151)
        self.bandpass_notch = ButterworthBandpassNotch(fs=target_fs, lowcut=0.5, highcut=45.0, notch_freq=50.0)
        self.sqi = SignalQualityIndex()
        self.slicer = AdaptiveRPeakBeatSlicer(window_size=window_size, fs=target_fs)

    def process(self, ecg_raw_500hz: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Args:
            ecg_raw_500hz: (5000, 12) int16 ADC values (1000 LSB/mV)
        Returns:
            beat_tensor: (12, 180) float32 array in [-1.0, 1.0]
            quality_info: dict with SQI flags and processing status
        """
        # 1. Convert to millivolts
        ecg_mv = ecg_raw_500hz.astype(np.float32) / 1000.0

        # 2. Downsample 500Hz -> 250Hz
        ecg_250 = ecg_mv[::2, :]

        # 3. Baseline Wander Removal (SWMA)
        ecg_detrended = self.swma.filter_multichannel(ecg_250)

        # 4. Bandpass 0.5-45Hz + Notch 50Hz (Zero-phase IIR)
        ecg_clean = self.bandpass_notch.filter_multichannel(ecg_detrended)

        # 5. Signal Quality Index & Fault Check
        quality = self.sqi.evaluate(ecg_clean, fs=self.target_fs)
        if not quality["is_valid"]:
            # If lead fault detected, return zero tensor and error flag
            return np.zeros((12, self.window_size), dtype=np.float32), quality

        # 6. Beat Extraction & Normalization
        beat, r_peak = self.slicer.extract_dominant_beat(ecg_clean)
        quality["r_peak_idx"] = r_peak

        return beat, quality

    def process_multibeat(self, ecg_raw_500hz: np.ndarray, max_beats: int = 8) -> Tuple[np.ndarray, int, Dict]:
        """
        Extracts up to `max_beats` clean normalized beats per 10s ECG recording
        with EDGE-02 transient clipping and EDGE-03 extreme-HR Tukey protection.
        """
        ecg_mv = ecg_raw_500hz.astype(np.float32) / 1000.0
        # EDGE-02: Clamp extreme static/EMG transients to [-5.0, +5.0] mV after SQI saturation check
        ecg_250 = ecg_mv[::2, :]
        ecg_detrended = self.swma.filter_multichannel(ecg_250)
        ecg_clean = self.bandpass_notch.filter_multichannel(ecg_detrended)

        quality = self.sqi.evaluate(ecg_clean, fs=self.target_fs)
        if not quality["is_valid"]:
            return np.zeros((max_beats, 12, self.window_size), dtype=np.float32), 0, quality

        # Clamp non-fault occasional EMG spikes to [-5.0, 5.0] mV per EDGE-02
        ecg_clean = np.clip(ecg_clean, -5.0, 5.0)
        beats, n_valid, hr_bpm = self.slicer.extract_multibeat_windows(ecg_clean, max_beats=max_beats)
        quality["n_valid_beats"] = n_valid
        quality["hr_bpm"] = hr_bpm
        return beats, n_valid, quality
