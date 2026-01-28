import torch
import torch.nn as nn


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
        # Compute cosine similarity: sum(input * target) along dimension -1
        cos_sim = torch.sum(input * target, dim=-1, keepdim=True)  # [B, 1]
        
        # Convert to cosine distance: 1 - cos_sim
        # Range: [0, 2] where 0 means identical, 2 means opposite
        cosine_dist = 1.0 - cos_sim  # [B, 1]
        
        # Gaussian NLL-inspired loss with cosine distance:
        # 0.5 * (dist / var + log(var))
        loss = 0.5 * (cosine_dist / (var + self.eps) + torch.log(var + self.eps))
        
        return loss.mean()
