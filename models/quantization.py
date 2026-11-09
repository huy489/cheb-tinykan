"""
Quantization-Aware Simulation, L1 Edge Pruning & C Header Exporter for Cheb-TinyKAN.
Implements:
1. Fixed-point Q1.15 representation & fake quantization for ARM Cortex-M4 CMSIS-NN.
2. L1 magnitude-based edge pruning (targeting >=70% sparsity on Cheb-KAN Bottleneck).
3. Automated C header code generation (`firmware/cheb_tinykan_weights.h`).

References:
- RES-01-KAN-TINYEDGE.md: Section 4.2 (Fixed-point Q1.15 arithmetic)
- PRD-01-CHEB-TINYKAN.md: AC-03 (<12K params), AC-07 (<1.5% Q1.15 error)
"""

import os
import math
import numpy as np


class Q15Quantizer:
    """
    Fixed-point Q1.15 Quantizer for 16-bit signed integer arithmetic.
    Scale: 2^15 = 32768.
    Range: [-1.0, 32767/32768] ~ [-32768, 32767].
    """
    SCALE = 32768.0
    MIN_INT16 = -32768
    MAX_INT16 = 32767

    @classmethod
    def quantize_to_int16(cls, x: np.ndarray) -> np.ndarray:
        """Converts float32 in [-1.0, 1.0] to int16 Q1.15 values."""
        scaled = np.round(x * cls.SCALE)
        clamped = np.clip(scaled, cls.MIN_INT16, cls.MAX_INT16)
        return clamped.astype(np.int16)

    @classmethod
    def dequantize(cls, q: np.ndarray) -> np.ndarray:
        """Converts int16 Q1.15 back to float32."""
        return q.astype(np.float32) / cls.SCALE

    @classmethod
    def fake_quantize(cls, x: np.ndarray) -> np.ndarray:
        """Simulates Q1.15 precision loss in floating point."""
        return cls.dequantize(cls.quantize_to_int16(x))


class KANEdgePruner:
    """
    L1 Magnitude Pruning Engine for Cheb-KAN layers.
    Prunes the weakest connections until the desired sparsity target is reached.
    """

    @staticmethod
    def prune_cheb_kan_numpy(
        base_weight: np.ndarray,
        cheb_coeffs: np.ndarray,
        target_sparsity: float = 0.70
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Prunes edges in numpy arrays.
        Args:
            base_weight: (out_features, in_features)
            cheb_coeffs: (out_features, in_features, degree + 1)
            target_sparsity: float, fraction of edges to prune (default 0.70 = 70%)
        Returns:
            pruned_base_weight, pruned_cheb_coeffs, mask, actual_sparsity
        """
        # Edge magnitude: L2 norm of Chebyshev coefficients + absolute base weight
        cheb_norm = np.linalg.norm(cheb_coeffs, axis=-1)  # (out_feat, in_feat)
        edge_energy = cheb_norm + np.abs(base_weight)

        # Determine threshold at the target percentile
        threshold = np.percentile(edge_energy, target_sparsity * 100.0)

        # Active mask: True = KEEP, False = PRUNE
        keep_mask = edge_energy >= threshold

        pruned_base = base_weight.copy()
        pruned_cheb = cheb_coeffs.copy()

        pruned_base[~keep_mask] = 0.0
        pruned_cheb[~keep_mask] = 0.0

        pruned_count = np.sum(~keep_mask)
        total_edges = keep_mask.size
        actual_sparsity = float(pruned_count) / float(total_edges)

        return pruned_base, pruned_cheb, keep_mask, actual_sparsity


class CHeaderExporter:
    """
    Exports trained and pruned weights into a standard C99 header file
    ready to compile with CMSIS-NN on ARM Cortex-M4 or ESP32-S3.
    """

    @staticmethod
    def export_weights_header(
        weights_dict: dict,
        output_path: str,
        quantize_q15: bool = True
    ) -> str:
        """
        Generates C arrays formatted for firmware consumption.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        lines = [
            "/*",
            " * CHEB-TINYKAN CMSIS-NN WEIGHT TABLE (AUTO-GENERATED)",
            " * Model: Cheb-TinyKAN (1D-CNN Tokenizer + Chebyshev KAN Bottleneck + 5-Head Sigmoid)",
            f" * Quantization: {'Fixed-Point Q1.15 (int16_t)' if quantize_q15 else 'Floating-Point (float32_t)'}",
            " * Target: ARM Cortex-M4 (STM32F411RE) & Xtensa LX7 (ESP32-S3)",
            " */",
            "",
            "#ifndef CHEB_TINYKAN_WEIGHTS_H",
            "#define CHEB_TINYKAN_WEIGHTS_H",
            "",
            "#include <stdint.h>",
            "",
            f"#define CHEB_KAN_Q15_SCALE {int(Q15Quantizer.SCALE)}",
            "#define CHEB_KAN_DEGREE 4",
            "#define CHEB_KAN_IN_FEATURES 64",
            "#define CHEB_KAN_BOTTLENECK_FEATURES 24",
            "#define CHEB_KAN_NUM_CLASSES 5",
            ""
        ]

        def format_array_1d(arr, name, c_type="int16_t"):
            lines.append(f"static const {c_type} {name}[{len(arr)}] = {{")
            chunk_size = 8
            for i in range(0, len(arr), chunk_size):
                chunk = arr[i:i + chunk_size]
                if c_type == "int16_t":
                    formatted = ", ".join(f"{int(v):>6}" for v in chunk)
                else:
                    formatted = ", ".join(f"{float(v):>9.5f}f" for v in chunk)
                sep = "," if (i + chunk_size < len(arr)) else ""
                lines.append(f"    {formatted}{sep}")
            lines.append("};\n")

        for key, val in weights_dict.items():
            arr = np.asarray(val)
            if quantize_q15:
                q_arr = Q15Quantizer.quantize_to_int16(arr).flatten()
                format_array_1d(q_arr, key, "int16_t")
            else:
                f_arr = arr.astype(np.float32).flatten()
                format_array_1d(f_arr, key, "float")

        lines.append("#endif /* CHEB_TINYKAN_WEIGHTS_H */\n")
        content = "\n".join(lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path
