"""
Multi-Label Focal Loss for Imbalanced Medical Diagnosis.
Resolves extreme class imbalance between common infarction sites (Anterior, Inferior)
and rare anatomical territories (Lateral, pure Septal).

Reference:
- Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
- PRD-01-CHEB-TINYKAN.md: Section 2.1 (Focal Loss with gamma=2.0, alpha=0.25).
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


if nn is not None:
    class MultiLabelFocalLoss(nn.Module):
        """
        Binary Focal Loss applied independently across multi-label output heads.
        Numerically stable formulation using BCEWithLogits.
        """
        def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """
            Args:
                logits: Tensor of shape (batch, num_classes) - unnormalized log-odds
                targets: Tensor of shape (batch, num_classes) - binary float {0.0, 1.0}
            Returns:
                loss: Scalar tensor
            """
            # Standard binary cross entropy with logits
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

            # Calculate p_t (probability of correct class)
            # When target=1, p_t = p = sigmoid(logits)
            # When target=0, p_t = 1 - p = 1 - sigmoid(logits)
            # Notice that exp(-bce_loss) is mathematically identical to p_t!
            p_t = torch.exp(-bce_loss)

            # Focal modulating factor: (1 - p_t)^gamma
            focal_weight = (1.0 - p_t) ** self.gamma

            # Alpha balancing factor
            if self.alpha is not None:
                alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
                focal_loss = alpha_t * focal_weight * bce_loss
            else:
                focal_loss = focal_weight * bce_loss

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
