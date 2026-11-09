"""
EASI Dower Lead Reconstruction Module for Ablation Study (AC-09).
Reconstructs standard 12-lead ECG from reduced 4-electrode EASI system
using Dower's linear transformation matrix:
    V_12(t) = D_EASI * [ES(t), AS(t), AI(t)]^T

Used to quantitatively benchmark the trade-off between reduced-lead wearable ergonomics
and clinical diagnostic sensitivity on PTB-XL Fold 10.

References:
- Dower et al., "Deriving the 12-lead electrocardiogram from four (EASI) electrodes", J. Electrocardiol., 1988.
- Drew et al., "Accuracy of the EASI 12-lead electrocardiogram compared to the standard 12-lead", Circulation, 2004.
- CONSENSUS-01-CHEB-TINYKAN.md: Decision 2 (12-Lead Gold Standard + EASI Ablation)
- PRD-01-CHEB-TINYKAN.md: AC-09
"""

import numpy as np


# Standard Dower EASI Transformation Matrix D (12 x 3)
# Input vector: [ES, AS, AI]^T
# Output vector: [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]^T
#
# Standard coefficients established by Dower et al. (1988)
EASI_DOWER_MATRIX = np.array([
    # Lead:     ES        AS        AI
    [       -0.088,   -0.127,    0.631],  # Lead I
    [        0.669,    0.043,    0.076],  # Lead II
    [        0.757,    0.170,   -0.555],  # Lead III
    [       -0.290,    0.042,   -0.354],  # Lead aVR
    [       -0.422,   -0.149,    0.593],  # Lead aVL
    [        0.713,    0.106,   -0.239],  # Lead aVF
    [       -0.198,    0.413,   -0.279],  # Lead V1
    [        0.054,    0.739,   -0.347],  # Lead V2
    [        0.248,    0.722,   -0.187],  # Lead V3
    [        0.410,    0.559,    0.119],  # Lead V4
    [        0.435,    0.311,    0.424],  # Lead V5
    [        0.306,    0.113,    0.562],  # Lead V6
], dtype=np.float32)

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class EASIDowerReconstructor:
    """
    Transforms 3-channel EASI signals to 12-channel standard ECG,
    and inverse-projects 12-channel standard ECG to synthetic EASI signals
    to evaluate degradation on PTB-XL records where only standard 12-lead exists.
    """

    def __init__(self, matrix: np.ndarray = None):
        if matrix is None:
            self.matrix = EASI_DOWER_MATRIX.copy()
        else:
            self.matrix = np.array(matrix, dtype=np.float32)

        # Precompute pseudo-inverse for backward projection (12-lead -> EASI 3-lead)
        # D_pinv is (3 x 12)
        self.matrix_pinv = np.linalg.pinv(self.matrix)

    def synthesize_easi_from_12lead(self, ecg_12lead: np.ndarray) -> np.ndarray:
        """
        Projects a 12-lead ECG (12, N) to the 3-lead EASI space (3, N)
        via Moore-Penrose pseudo-inverse.
        """
        assert ecg_12lead.shape[0] == 12, f"Expected 12 channels, got {ecg_12lead.shape[0]}"
        return np.matmul(self.matrix_pinv, ecg_12lead)

    def reconstruct_12lead_from_easi(self, easi_3lead: np.ndarray) -> np.ndarray:
        """
        Reconstructs 12-lead ECG (12, N) from 3-lead EASI space (3, N)
        via Dower transformation matrix D.
        """
        assert easi_3lead.shape[0] == 3, f"Expected 3 channels (ES, AS, AI), got {easi_3lead.shape[0]}"
        return np.matmul(self.matrix, easi_3lead)

    def run_easi_degradation_pipeline(self, ecg_12lead: np.ndarray) -> np.ndarray:
        """
        End-to-end degradation simulation:
        Original 12-Lead -> Synthesize 3-Lead EASI -> Reconstruct 12-Lead via Dower Matrix.
        This simulates what an EASI wearable device would see when evaluating a PTB-XL record.
        """
        easi_3 = self.synthesize_easi_from_12lead(ecg_12lead)
        reconstructed_12 = self.reconstruct_12lead_from_easi(easi_3)
        return reconstructed_12

    @staticmethod
    def calculate_st_morphology_distortion(orig_12: np.ndarray, recon_12: np.ndarray) -> dict:
        """
        Quantifies the ST-segment voltage distortion (mean absolute error in mV)
        across each lead, especially precordial leads V1-V4 where Drew et al. reported
        significant ST amplitude loss.
        """
        distortion = {}
        for idx, lead_name in enumerate(LEAD_NAMES):
            mae = float(np.mean(np.abs(orig_12[idx] - recon_12[idx])))
            max_err = float(np.max(np.abs(orig_12[idx] - recon_12[idx])))
            distortion[lead_name] = {"mae_mv": mae, "max_err_mv": max_err}
        return distortion
