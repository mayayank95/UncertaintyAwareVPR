import logging
import numpy as np
import torch
from scipy.stats import pearsonr
from utils.util import cosine_distance
from pathlib import Path

logger = logging.getLogger(__name__)

def _compute_correlation(distances, variances):
    """Compute Pearson correlation between distances and variances."""
    if len(distances) > 1 and np.std(distances) > 0 and np.std(variances) > 0:
        corr, _ = pearsonr(distances, variances)
        return corr
    return 0.0

def compute_uncertainty_correlation(args, all_descriptors, all_variances, positives_per_query, num_database):
    """
    Computes the correlation between the uncertainty (variance) and the distance to the ground truth.
    """
    uncertainty_corr = 0.0
    if args['model_mode'] == "uncertainty" and args['use_labels'] and args['datasets_type'] == ['test']:
        logger.info("Computing uncertainty correlation metrics...")
        loss_type = args.get('uncertainty_loss', 'gaussian_nll').lower()
        
        if positives_per_query is None:
            logger.warning("Positives per query is None, cannot compute correlation.")
            return 0.0
        
        # Get query indices and their first positive ground truth from DB
        # Filter out queries with no positives
        valid_queries = [(i, pos[0]) for i, pos in enumerate(positives_per_query) if len(pos) > 0]
        
        if len(valid_queries) > 0:
            q_indices = np.array([i for i, _ in valid_queries])
            db_gt_indices = np.array([idx for _, idx in valid_queries])
            
            # Convert to tensors for normalization/distance math
            q_tensor = torch.from_numpy(all_descriptors[num_database + q_indices])
            db_tensor = torch.from_numpy(all_descriptors[db_gt_indices])
            q_var_tensor = torch.from_numpy(all_variances[num_database + q_indices])

            q_norm = torch.nn.functional.normalize(q_tensor, p=2, dim=1)
            db_norm = torch.nn.functional.normalize(db_tensor, p=2, dim=1)

            if loss_type == 'gaussian_cosine':
                dists = cosine_distance(q_norm, db_norm)
            else:
                dists = torch.sum((q_norm - db_norm) ** 2, dim=-1)      
            mean_vars = torch.mean(q_var_tensor, dim=-1)
            uncertainty_corr = _compute_correlation(dists.numpy(), mean_vars.numpy())

    return uncertainty_corr

def normalize_variance(variances: np.ndarray, method: str) -> np.ndarray:
    """Normalize per-query mean variance.

    Args:
        variances: [num_queries, D] raw variance vectors.
        method: 'minmax' scales to [0, 1], 'zscore' standardizes to mean=0 / std=1.

    Returns:
        [num_queries, D] normalized variance vectors.
    """
    mean_var = np.mean(variances, axis=-1, keepdims=True)
    if method == "minmax":
        v_min, v_max = mean_var.min(), mean_var.max()
        if v_max - v_min < 1e-12:
            logger.warning("Variance range is near-zero; skipping minmax normalization.")
            return variances
        scale = (mean_var - v_min) / (v_max - v_min)
    elif method == "zscore":
        v_std = mean_var.std()
        if v_std < 1e-12:
            logger.warning("Variance std is near-zero; skipping zscore normalization.")
            return variances
        scale = (mean_var - mean_var.mean()) / v_std
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    logger.info(f"Variance normalized ({method}): range [{scale.min():.4f}, {scale.max():.4f}]")
    return variances * (scale / (mean_var + 1e-12))


def compute_uncertainty_statistics(all_variances, output_dir=None):
    """
    Computes and logs statistics about the uncertainty values.
    """
    mean_var = np.mean(all_variances)
    min_var = np.min(all_variances)
    max_var = np.max(all_variances)
    std_var = np.std(all_variances)
    median_var = np.median(all_variances)
    
    logger.info(
        f"Variance Statistics - Mean: {mean_var:.4e}, Min: {min_var:.4e}, Max: {max_var:.4e}, Std: {std_var:.4e}, Median: {median_var:.4e}"
    )

    if output_dir:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.hist(all_variances.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            plt.title("Histogram of Uncertainty Values (Variances)")
            plt.xlabel("Variance")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.savefig(Path(output_dir) / "uncertainty_histogram.png")
            plt.close()
        except ImportError:
            logger.warning("matplotlib not installed, skipping uncertainty histogram.")

    return {
        "mean": mean_var,
        "min": min_var,
        "max": max_var,
        "std": std_var,
        "median": median_var
    }