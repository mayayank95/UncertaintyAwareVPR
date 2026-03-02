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
    Pearson correlation between per-query variance and distance from query to its nearest positive.
    For VPR: well-calibrated uncertainty should be higher when the match is hard (large distance)
    and lower when easy (small distance). Positive correlation = variance tracks retrieval difficulty.
    """
    uncertainty_corr = 0.0
    if args["model_mode"] == "uncertainty" and args["use_labels"]:
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


def plot_variance_distribution(all_variances, output_dir, num_database=None):
    """
    Save a figure with variance distribution: (1) all variance values, (2) per-vector mean variance (db vs query if num_database given).
    """
    try:
        import matplotlib.pyplot as plt
        out_path = Path(output_dir)
        # (1) All variance elements
        flat = all_variances.flatten()
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].hist(flat, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axs[0].set_title("Distribution of variance (all elements)")
        axs[0].set_xlabel("Variance σ²")
        axs[0].set_ylabel("Frequency")
        axs[0].grid(True, alpha=0.3)

        # (2) Per-vector mean variance
        mean_per_vector = np.mean(all_variances, axis=1)
        if num_database is not None and num_database < len(mean_per_vector):
            db_means = mean_per_vector[:num_database]
            q_means = mean_per_vector[num_database:]
            axs[1].hist(db_means, bins=40, alpha=0.6, color='green', label='Database', edgecolor='black')
            axs[1].hist(q_means, bins=40, alpha=0.6, color='coral', label='Queries', edgecolor='black')
            axs[1].set_title("Per-vector mean variance (database vs queries)")
        else:
            axs[1].hist(mean_per_vector, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            axs[1].set_title("Per-vector mean variance")
        axs[1].set_xlabel("Mean variance σ²")
        axs[1].set_ylabel("Frequency")
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path / "variance_distribution.png", dpi=150)
        plt.close()
        logger.info(f"Variance distribution plot saved to {out_path / 'variance_distribution.png'}")
    except ImportError:
        logger.warning("matplotlib not installed, skipping variance distribution plot.")


def compute_uncertainty_statistics(all_variances, output_dir=None, num_database=None):
    """
    Computes and logs statistics about the uncertainty values, and saves variance distribution plot.
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
        plot_variance_distribution(all_variances, output_dir, num_database=num_database)

    return {
        "mean": mean_var,
        "min": min_var,
        "max": max_var,
        "std": std_var,
        "median": median_var
    }