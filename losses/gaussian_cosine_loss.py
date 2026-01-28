import torch
import torch.nn as nn
from losses.cosface_loss import cosine_distance


class GaussianCosineLoss(nn.Module):
    """
    Gaussian Cosine Loss for uncertainty-aware learning.
    
    Combines cosine distance with uncertainty estimation to provide a probabilistic
    measure of similarity between embeddings and their class prototypes.
    
    The loss is computed as:
        0.5 * (dist / var + log(var))
    
    where dist is the cosine distance (1 - cosine_similarity) and var is the predicted variance.
    
    Args:
        eps (float): Small epsilon value to prevent numerical instability. Default: 1e-6
    """
    
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    @staticmethod
    def compute_cosine_distance(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine distance between input and target vectors.
        
        Args:
            input (torch.Tensor): Normalized embedding vectors of shape [B, D]
            target (torch.Tensor): Target/prototype vectors of shape [B, D]
        
        Returns:
            torch.Tensor: Cosine distance of shape [B], range [0, 2]
        """
        return cosine_distance(input, target)

    def forward(self, input: torch.Tensor, target: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian Cosine Loss.
        
        Args:
            input (torch.Tensor): Normalized embedding vectors of shape [B, D]
            target (torch.Tensor): Target/prototype vectors of shape [B, D]
            var (torch.Tensor): Predicted variance values of shape [B, D]
        
        Returns:
            torch.Tensor: Scalar loss value (mean over batch and dimensions)
        """
        # Compute cosine distance
        cosine_dist = self.compute_cosine_distance(input, target)  # [B]
        
        # Gaussian NLL-inspired loss with cosine distance:
        # 0.5 * (dist / var + log(var))
        # Average variance across dimensions
        mean_var = torch.mean(var, dim=-1)  # [B]
        loss = 0.5 * (cosine_dist / (mean_var + self.eps) + torch.log(mean_var + self.eps))
        
        return loss.mean()
