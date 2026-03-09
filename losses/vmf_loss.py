import torch
import torch.nn.functional as F
import math
from scipy.special import ive as scipy_ive

class VMFLikelihood(torch.nn.Module):
    def __init__(self, d=256, eps=1e-7):
        super().__init__()
        self.d = d
        self.eps = eps

    @staticmethod
    def _scipy_log_ive(v, kappa):
        """Compute log(I_v(kappa) * e^{-kappa}) + kappa = log(I_v(kappa))
        using scipy for general order v. Returns a tensor on the same device."""
        kappa_np = kappa.detach().cpu().numpy()
        ive_vals = scipy_ive(v, kappa_np)
        # Clamp to avoid log(0)
        ive_vals[ive_vals <= 0] = 1e-30
        log_ive = torch.tensor(ive_vals, dtype=kappa.dtype, device=kappa.device).log()
        return log_ive + kappa  # log(I_v(kappa))

    def log_bessel_approx(self, kappa):
        """
        Approximates log(Z_d(kappa)) for high dimensions.
        Uses the approximation: 
        log Z_d(kappa) ≈ (d-1)/2 * log(kappa) - kappa + constant
        Note: In most training scenarios, we only need terms involving kappa.
        """
        # A more robust approximation for log-partition function in high-D:
        # log Z_d(kappa) = kappa * sqrt(1 + (d/kappa)^2) ... (asymptotic)
        # For training, we often use the simplified form derived from 
        # the saddle-point approximation.
        
        v = (self.d - 1) / 2
        # Use log1p for stability if kappa is near zero
        #return v * torch.log(kappa + self.eps) - kappa 

        # Asymptotic approximation (accurate for large kappa):
        asymptotic = kappa + v * torch.log(kappa) - 0.5 * math.log(2 * math.pi) - 0.5 * torch.log(kappa)
        # Exact computation via scipy (for small kappa):
        exact = self._scipy_log_ive(v, kappa)

        # A stable implementation for training:
        log_iv = torch.where(
            kappa > 10, 
            asymptotic,
            exact
        )

        return v * torch.log(kappa + self.eps) - (self.d / 2) * torch.log(torch.tensor(2 * torch.pi)) - log_iv



    def forward(self, mu, kappa, target):
        """
        mu: [batch, d] - Predicted mean (must be L2 normalized)
        kappa: [batch, 1] - Predicted concentration (from Softplus)
        target: [batch, d] - Ground truth vector (must be L2 normalized)
        """
        # 1. Ensure inputs are normalized (defensive)
        mu = F.normalize(mu, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)
        
        # 2. Dot product (Cosine Similarity)
        # We want to maximize kappa * dot_product
        dot_prod = torch.sum(mu * target, dim=-1, keepdim=True)
        
        # 3. vMF Log-Likelihood
        # NLL = - (kappa * cos_sim - log_partition)
        # We simplify the normalization constant for the loss
        log_partition = self.log_bessel_approx(kappa)
        
        loss = -(kappa * dot_prod + log_partition)
        
        return loss.mean()