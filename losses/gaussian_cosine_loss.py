import torch
import torch.nn as nn
from utils.util import cosine_distance


class GaussianCosineLoss(nn.Module):
    """Gaussian NLL-inspired loss using cosine distance instead of per-element L2.

    Loss = 0.5 * (d_cos / var + log(var))

    Equivalent to GaussianNLLLoss with uniform variance on L2-normalized vectors
    (since ||a-b||² = 2 * d_cos for unit vectors), but avoids the per-element
    scaling mismatch that makes standard GaussianNLLLoss ignore the distance term.

    Args:
        eps: Small value to prevent division by zero and log(0).
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: L2-normalized embeddings [B, D].
            target: L2-normalized prototype vectors [B, D].
            var: Predicted variance [B, D] (averaged to scalar per sample).

        Returns:
            Scalar loss (mean over batch).
        """
        dist = cosine_distance(input, target)           # [B]
        mean_var = torch.mean(var, dim=-1)               # [B]
        loss = 0.5 * (dist / (mean_var + self.eps) + torch.log(mean_var + self.eps))
        return loss.mean()
