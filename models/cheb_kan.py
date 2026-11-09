"""
Cheb-TinyKAN: Orthogonal Polynomial Kolmogorov-Arnold Network Layer.
Mathematically derived for TinyML deployment on ARM Cortex-M4 & ESP32-S3.

Reference:
- Baghershahi et al., "Cheb-KAN: Orthogonal Polynomial KAN", 2024.
- RES-01-KAN-TINYEDGE.md: Section 4.2 (Three-term Recurrence & Fixed-Point Q1.15).
"""

import math
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


class ChebyshevPolynomial:
    """
    Mathematical definition and recursive evaluation of Chebyshev polynomials of the 1st kind.
    Enforces strict 3-term recurrence:
        T_0(x) = 1
        T_1(x) = x
        T_{k+1}(x) = 2x * T_k(x) - T_{k-1}(x)
    """

    @staticmethod
    def forward_numpy(x, degree=4):
        """
        Numpy implementation for local verification without PyTorch.
        Args:
            x: ndarray of shape (..., ) with values bounded in [-1.0, 1.0]
            degree: int, polynomial degree (default 4)
        Returns:
            T: ndarray of shape (..., degree + 1)
        """
        import numpy as np
        x_clamped = np.clip(x, -1.0, 1.0)
        shape = list(x_clamped.shape) + [degree + 1]
        T = np.zeros(shape, dtype=np.float32)

        T[..., 0] = 1.0
        if degree >= 1:
            T[..., 1] = x_clamped
        for k in range(1, degree):
            T[..., k + 1] = 2.0 * x_clamped * T[..., k] - T[..., k - 1]
        return T

    @staticmethod
    def forward_torch(x, degree=4):
        """
        PyTorch implementation vectorized over batches and features.
        Args:
            x: Tensor of shape (batch, in_features)
            degree: int, polynomial degree
        Returns:
            T: Tensor of shape (batch, in_features, degree + 1)
        """
        x_clamped = torch.clamp(x, -1.0, 1.0)
        batch_size, in_features = x_clamped.shape
        T = torch.empty((batch_size, in_features, degree + 1), device=x.device, dtype=x.dtype)

        T[:, :, 0] = 1.0
        if degree >= 1:
            T[:, :, 1] = x_clamped
        for k in range(1, degree):
            T[:, :, k + 1] = 2.0 * x_clamped * T[:, :, k] - T[:, :, k - 1]
        return T


if nn is not None:
    class ChebyshevKANLayer(nn.Module):
        """
        A single KAN layer with orthogonal Chebyshev polynomial edge activations.
        Each edge activation is parameterized as:
            phi_{j, i}(x) = w_base^{(j, i)} * tanh(x) + sum_{k=0}^P c_k^{(j, i)} * T_k(x)
        """

        def __init__(self, in_features: int, out_features: int, degree: int = 4, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.degree = degree

            # Learnable base linear activation weights: out_features x in_features
            self.base_weight = nn.Parameter(torch.empty(out_features, in_features))

            # Learnable Chebyshev polynomial coefficients: out_features x in_features x (degree + 1)
            self.cheb_coeffs = nn.Parameter(torch.empty(out_features, in_features, degree + 1))

            if bias:
                self.bias = nn.Parameter(torch.empty(out_features))
            else:
                self.register_parameter('bias', None)

            self.reset_parameters()

        def reset_parameters(self):
            # Base weight initialization (Xavier uniform scaled for tanh)
            nn.init.xavier_uniform_(self.base_weight, gain=1.0)

            # Chebyshev coefficients initialized with decaying amplitude for higher degrees
            # to guarantee convergence and prevent high-frequency oscillations
            with torch.no_grad():
                for k in range(self.degree + 1):
                    scale = 1.0 / (k + 1.0)
                    bound = scale * (1.0 / math.sqrt(self.in_features))
                    self.cheb_coeffs[:, :, k].uniform_(-bound, bound)

            if self.bias is not None:
                nn.init.zeros_(self.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Tensor of shape (batch, in_features), bounded or unnormalized
            Returns:
                out: Tensor of shape (batch, out_features)
            """
            # 1. Base activation term: w_base * tanh(x)
            base_out = F.linear(torch.tanh(x), self.base_weight)

            # 2. Chebyshev polynomial recurrence term: sum(c_k * T_k(x))
            # x is clamped to [-1.0, 1.0] inside ChebyshevPolynomial.forward_torch
            T = ChebyshevPolynomial.forward_torch(x, degree=self.degree)  # (B, in_feat, degree + 1)

            # Compute tensor contraction: (B, in_feat, deg+1) x (out_feat, in_feat, deg+1) -> (B, out_feat)
            cheb_out = torch.einsum('bid,oid->bo', T, self.cheb_coeffs)

            # 3. Aggregate
            out = base_out + cheb_out
            if self.bias is not None:
                out = out + self.bias
            return out

        def l1_regularization(self) -> torch.Tensor:
            """
            Calculates L1 penalty on Chebyshev polynomial coefficients to enforce edge sparsity.
            Prunes weak connections (aiming for >= 70% pruned edges in Cheb-KAN layer 1).
            """
            return torch.mean(torch.abs(self.cheb_coeffs))

        def prune_edges(self, threshold: float = 1e-3) -> int:
            """
            Hard-prunes edges whose L2 coefficient energy is below threshold.
            Sets their weights and coefficients to exactly 0.0.
            Returns number of pruned edges.
            """
            with torch.no_grad():
                edge_norm = torch.norm(self.cheb_coeffs, dim=-1)  # (out_feat, in_feat)
                mask = edge_norm < threshold
                self.cheb_coeffs[mask] = 0.0
                self.base_weight[mask] = 0.0
                return int(mask.sum().item())

        def export_symbolic_polynomials(self):
            """
            Extracts canonical closed-form polynomials for clinical interpretation:
                phi(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
            Uses linear matrix transform M connecting Chebyshev coefficients c_k to monomial coefficients a_k.
            """
            if self.degree != 4:
                raise NotImplementedError("Closed-form extraction currently implemented for degree P=4.")

            # Transformation matrix for degree 4
            # [a0, a1, a2, a3, a4]^T = M * [c0, c1, c2, c3, c4]^T
            M = torch.tensor([
                [ 1.0,  0.0, -1.0,  0.0,  1.0],
                [ 0.0,  1.0,  0.0, -3.0,  0.0],
                [ 0.0,  0.0,  2.0,  0.0, -8.0],
                [ 0.0,  0.0,  0.0,  4.0,  0.0],
                [ 0.0,  0.0,  0.0,  0.0,  8.0]
            ], device=self.cheb_coeffs.device)

            with torch.no_grad():
                # self.cheb_coeffs is (out_features, in_features, 5)
                # canonical_coeffs will be (out_features, in_features, 5)
                canonical_coeffs = torch.einsum('ij,omj->omi', M, self.cheb_coeffs)
            return canonical_coeffs
else:
    ChebyshevKANLayer = None

