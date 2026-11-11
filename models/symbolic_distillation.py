"""
Symbolic Morphometric Distillation Module for Cheb-TinyKAN.
Converts learned Chebyshev polynomial edge activations into canonical closed-form
monomial polynomials:
    phi(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4

Provides clinical explainability for cardiologists and bedside emergency alerts
(mapping Lead V1-V4 to LAD/AMI, II/III/aVF to RCA/IMI, I/aVL/V5-V6 to LCx/LMI).

References:
- CONSENSUS-01-CHEB-TINYKAN.md: Decision 1 (Clinical Symbolic Morphometry)
- PRD-01-CHEB-TINYKAN.md: AC-06, AC-07
"""

import numpy as np


# Transformation matrix M connecting Chebyshev coefficients c = [c0, c1, c2, c3, c4]^T
# to canonical monomial coefficients a = [a0, a1, a2, a3, a4]^T
# Based on:
# T_0(x) = 1
# T_1(x) = x
# T_2(x) = 2x^2 - 1
# T_3(x) = 4x^3 - 3x
# T_4(x) = 8x^4 - 8x^2 + 1
#
# a0 = c0 - c2 + c4
# a1 = c1 - 3*c3
# a2 = 2*c2 - 8*c4
# a3 = 4*c3
# a4 = 8*c4
CHEBYSHEV_TO_CANONICAL_MATRIX_DEGREE_4 = np.array([
    [ 1.0,  0.0, -1.0,  0.0,  1.0],
    [ 0.0,  1.0,  0.0, -3.0,  0.0],
    [ 0.0,  0.0,  2.0,  0.0, -8.0],
    [ 0.0,  0.0,  0.0,  4.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  8.0]
], dtype=np.float32)

# Clinical territory descriptions
TERRITORY_METADATA = {
    "AMI": {
        "name": "Anterior Myocardial Infarction",
        "artery": "Left Anterior Descending (LAD)",
        "primary_leads": ["V1", "V2", "V3", "V4"],
        "reciprocal_leads": ["II", "III", "aVF"],
        "urgency": "CRITICAL - Immediate Cath-Lab Activation Required"
    },
    "IMI": {
        "name": "Inferior Myocardial Infarction",
        "artery": "Right Coronary Artery (RCA) / Left Circumflex (LCx)",
        "primary_leads": ["II", "III", "aVF"],
        "reciprocal_leads": ["I", "aVL"],
        "urgency": "HIGH - Monitor for Bradycardia and AV Block"
    },
    "LMI": {
        "name": "Lateral Myocardial Infarction",
        "artery": "Left Circumflex (LCx) / Diagonal LAD",
        "primary_leads": ["I", "aVL", "V5", "V6"],
        "reciprocal_leads": ["II", "III", "aVF"],
        "urgency": "HIGH - Risk of Left Ventricular Dysfunction"
    },
    "SMI": {
        "name": "Septal Myocardial Infarction",
        "artery": "Septal Perforator Branches of LAD",
        "primary_leads": ["V1", "V2"],
        "reciprocal_leads": ["None prominent"],
        "urgency": "HIGH - Risk of Ventricular Septal Defect"
    },
    "NORM": {
        "name": "Normal Sinus Rhythm",
        "artery": "None (Hemodynamically Stable)",
        "primary_leads": [],
        "reciprocal_leads": [],
        "urgency": "LOW - Routine ECG Monitoring"
    }
}


class SymbolicMorphometricDistiller:
    """
    Translates Cheb-TinyKAN weights into human-readable algebraic formulas
    and clinical diagnostic reports.
    """

    @staticmethod
    def chebyshev_to_canonical(cheb_coeffs: np.ndarray) -> np.ndarray:
        """
        Converts Chebyshev coefficients (..., 5) to monomial coefficients (..., 5).
        """
        assert cheb_coeffs.shape[-1] == 5, f"Expected last dimension 5 for degree 4, got {cheb_coeffs.shape[-1]}"
        return np.matmul(cheb_coeffs, CHEBYSHEV_TO_CANONICAL_MATRIX_DEGREE_4.T)

    @staticmethod
    def format_polynomial(a_coeffs: np.ndarray, variable_name: str = "x", threshold: float = 1e-3) -> str:
        """
        Formats coefficients [a0, a1, a2, a3, a4] as a readable string:
            a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
        Omits terms where |a_k| < threshold.
        """
        terms = []
        for k, a in enumerate(a_coeffs):
            if abs(a) < threshold:
                continue
            if k == 0:
                terms.append(f"{a:+.3f}")
            elif k == 1:
                terms.append(f"{a:+.3f}*{variable_name}")
            else:
                terms.append(f"{a:+.3f}*{variable_name}^{k}")
        if not terms:
            return "0.0"
        res = " ".join(terms)
        if res.startswith("+"):
            res = res[1:].strip()
        return res

    @staticmethod
    def generate_clinical_alert(
        territory: str,
        probability: float,
        lead_elevations: dict,
        polynomial_str: str = ""
    ) -> str:
        """
        Generates a standardized clinical bedside alert string suitable for
        OLED display or BLE transmission.
        """
        meta = TERRITORY_METADATA.get(territory, TERRITORY_METADATA["NORM"])
        lines = [
            "=" * 60,
            f"🚨 CLINICAL ALERT: {meta['name'].upper()} ({territory})",
            "=" * 60,
            f"- Detection Probability: {probability * 100:.1f}%",
            f"- Suspected Culprit Artery: {meta['artery']}",
            f"- Clinical Protocol: {meta['urgency']}",
            "-" * 60,
            "🔍 ST-SEGMENT & MORPHOMETRIC EVIDENCE:"
        ]

        for lead, val in lead_elevations.items():
            status = "ST-Elevation" if val > 0.10 else ("ST-Depression" if val < -0.05 else "Normal")
            lines.append(f"  + Lead {lead:>3}: {val:+.2f} mV ({status})")

        if polynomial_str:
            lines.append("-" * 60)
            lines.append("📐 ALGEBRAIC KAN EXPLANATION:")
            lines.append(f"  Risk({territory}) ≈ σ( {polynomial_str} )")

        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def classify_clinical_state(probabilities, thresholds=None) -> dict:
        """
        Implements PRD-01 EDGE-05 borderline monitoring band:
        - MI prob > 0.65 (and above optional class threshold) -> CRITICAL_MI_ALERT
        - 0.45 <= MI prob <= 0.65 -> SUSPECTED_ISCHEMIA_MONITOR_TREND (15-min dynamic monitoring)
        - Otherwise -> NORMAL_SINUS_MONITORING
        Args:
            probabilities: sequence of length 5 corresponding to [NORM, AMI, IMI, LMI, SMI]
        """
        names = ["NORM", "AMI", "IMI", "LMI", "SMI"]
        probs = [float(p) for p in probabilities]
        mi_pairs = [(names[i], probs[i]) for i in range(1, 5)]
        top_mi_name, top_mi_prob = max(mi_pairs, key=lambda kv: kv[1])

        if thresholds is not None and len(thresholds) == 5:
            high_cut = max(0.65, float(thresholds[names.index(top_mi_name)]))
        else:
            high_cut = 0.65

        if top_mi_prob > high_cut:
            state = "CRITICAL_MI_ALERT"
            action = "Immediate Cath-Lab triage & symbolic ST inspection"
        elif any(0.45 <= p <= 0.65 for _, p in mi_pairs):
            state = "SUSPECTED_ISCHEMIA_MONITOR_TREND"
            action = "Monitor ST-segment dynamics for next 15 minutes (EDGE-05)"
        else:
            state = "NORMAL_SINUS_MONITORING"
            action = "Routine wearable telemetry"

        return {
            "clinical_state": state,
            "top_mi_territory": top_mi_name,
            "top_mi_prob": top_mi_prob,
            "recommended_action": action,
        }
