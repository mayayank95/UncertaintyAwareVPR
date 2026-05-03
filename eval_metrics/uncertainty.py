import logging
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve, auc
from losses.gaussian_cosine_loss import cosine_distance
from pathlib import Path

logger = logging.getLogger(__name__)

def _compute_correlation(distances, variances):
    """Compute Spearman correlation between distances and variances."""
    if len(distances) > 1 and np.std(distances) > 0 and np.std(variances) > 0:
        corr, _ = spearmanr(distances, variances)
        return float(corr) if not np.isnan(corr) else 0.0
    return 0.0

# def compute_uncertainty_correlation(args, q_desc, db_desc, q_var, positives_per_query, output_dir=None):
#     """
#     Spearman correlation between per-query variance and distance from query to its nearest positive.
#     For VPR: well-calibrated uncertainty should be higher when the match is hard (large distance)
#     and lower when easy (small distance). Positive correlation = variance tracks retrieval difficulty.
#
#     If output_dir is provided, saves a scatter plot of distance vs mean variance.
#     """
#     uncertainty_corr = 0.0
#     if args["model_mode"] == "uncertainty" and args["use_labels"]:
#         logger.debug("Computing uncertainty correlation metrics...")
#         loss_type = args.get('uncertainty_loss', 'gaussian_nll').lower()
#
#         if positives_per_query is None:
#             logger.warning("Positives per query is None, cannot compute correlation.")
#             return 0.0
#
#         valid_queries = [(i, pos[0]) for i, pos in enumerate(positives_per_query) if len(pos) > 0]
#
#         if len(valid_queries) > 0:
#             q_indices = np.array([i for i, _ in valid_queries])
#             db_gt_indices = np.array([idx for _, idx in valid_queries])
#
#             q_tensor = torch.from_numpy(q_desc[q_indices])
#             db_tensor = torch.from_numpy(db_desc[db_gt_indices])
#             q_var_tensor = torch.from_numpy(q_var[q_indices])
#
#             q_norm = torch.nn.functional.normalize(q_tensor, p=2, dim=1)
#             db_norm = torch.nn.functional.normalize(db_tensor, p=2, dim=1)
#
#             if loss_type in ('gaussian_cosine', 'vmf'):
#                 dists = cosine_distance(q_norm, db_norm)
#             else:
#                 dists = torch.sum((q_norm - db_norm) ** 2, dim=-1)
#
#             mean_vars = torch.mean(q_var_tensor, dim=-1)
#
#             # For vMF, mean_vars is concentration (kappa); invert so high value = high uncertainty.
#             if loss_type == 'vmf':
#                 mean_vars = 1.0 / (mean_vars + 1e-6)
#
#             dists_np = dists.detach().cpu().numpy()
#             mean_vars_np = mean_vars.detach().cpu().numpy()
#             uncertainty_corr = _compute_correlation(dists_np, mean_vars_np)
#
#             if output_dir is not None:
#                 try:
#                     import matplotlib.pyplot as plt
#                     out_path = Path(output_dir).resolve()
#                     out_path.mkdir(parents=True, exist_ok=True)
#
#                     fig, ax = plt.subplots(figsize=(8, 6))
#                     ax.scatter(dists_np, mean_vars_np, alpha=0.4, s=18, color='steelblue', edgecolors='none')
#
#                     if len(dists_np) > 1:
#                         coeffs = np.polyfit(dists_np, mean_vars_np, 1)
#                         x_line = np.linspace(dists_np.min(), dists_np.max(), 100)
#                         ax.plot(x_line, np.polyval(coeffs, x_line), color='tomato', linewidth=2, label='Linear fit')
#
#                     dist_label = 'Cosine distance' if loss_type in ('gaussian_cosine', 'vmf') else 'L2 distance²'
#                     ax.set_xlabel(dist_label, fontsize=12)
#                     ax.set_ylabel('Uncertainty', fontsize=12)
#                     ax.set_title(f'Uncertainty vs Retrieval Distance  (Spearman ρ = {uncertainty_corr:.3f})', fontsize=13)
#                     ax.legend(fontsize=10)
#                     ax.grid(True, alpha=0.3)
#
#                     plt.tight_layout()
#                     save_path = out_path / 'uncertainty_correlation_scatter.png'
#                     plt.savefig(save_path, dpi=150)
#                     plt.close(fig)
#                     logger.debug(f"Uncertainty correlation scatter plot saved to {save_path}")
#                 except ImportError:
#                     logger.warning("matplotlib not installed, skipping uncertainty correlation scatter plot.")
#
#     return uncertainty_corr


def plot_variance_distribution(all_variances, output_dir, num_database=None):
    """
    Save a figure with per-vector mean variance: left = Database, right = Queries.

    Fallback when `num_database` is not provided (or invalid): single panel over
    all vectors combined.
    """
    try:
        import matplotlib.pyplot as plt
        out_path = Path(output_dir).resolve()
        if out_path.exists() and not out_path.is_dir():
            logger.warning("Cannot save variance distribution: %s exists as a file, not a directory.", out_path)
            return
        out_path.mkdir(parents=True, exist_ok=True)

        mean_per_vector = np.mean(all_variances, axis=1)
        has_split = num_database is not None and 0 < num_database < len(mean_per_vector)

        if has_split:
            db_means = mean_per_vector[:num_database]
            q_means = mean_per_vector[num_database:]
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].hist(db_means, bins=40, alpha=0.8, color='green', edgecolor='black')
            axs[0].set_title(f"Database per-vector mean variance (N={len(db_means)})")
            axs[0].set_xlabel("Mean variance σ²")
            axs[0].set_ylabel("Frequency")
            axs[0].grid(True, alpha=0.3)

            axs[1].hist(q_means, bins=40, alpha=0.8, color='coral', edgecolor='black')
            axs[1].set_title(f"Queries per-vector mean variance (N={len(q_means)})")
            axs[1].set_xlabel("Mean variance σ²")
            axs[1].set_ylabel("Frequency")
            axs[1].grid(True, alpha=0.3)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.hist(mean_per_vector, bins=50, alpha=0.8, color='steelblue', edgecolor='black')
            ax.set_title(f"Per-vector mean variance (N={len(mean_per_vector)})")
            ax.set_xlabel("Mean variance σ²")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path / "variance_distribution.png", dpi=150)
        plt.close()
        logger.debug(f"Variance distribution plot saved to {out_path / 'variance_distribution.png'}")
    except ImportError:
        logger.warning("matplotlib not installed, skipping variance distribution plot.")
    except Exception as e:
        logger.warning("Error saving variance distribution plot: %s", e)

    # Return statistics as a dictionary
    stats = {
        "all_mean": float(np.mean(all_variances)),
        "all_std": float(np.std(all_variances)),
    }
    mean_per_vector = np.mean(all_variances, axis=1)
    if num_database is not None and num_database < len(mean_per_vector):
        db_means = mean_per_vector[:num_database]
        q_means = mean_per_vector[num_database:]
        stats.update({
            "db_mean": float(np.mean(db_means)),
            "db_std": float(np.std(db_means)),
            "q_mean": float(np.mean(q_means)),
            "q_std": float(np.std(q_means)),
            "q_min": float(np.min(q_means)),
            "q_max": float(np.max(q_means)),
        })
    else:
        stats.update({
            "mean": float(np.mean(mean_per_vector)),
            "std": float(np.std(mean_per_vector)),
            "min": float(np.min(mean_per_vector)),
            "max": float(np.max(mean_per_vector)),
        })
    return stats


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
        stats = plot_variance_distribution(all_variances, output_dir, num_database=num_database)
    else:
        # If no plot, still compute the stats
        stats = {
            "all_mean": float(np.mean(all_variances)),
            "all_std": float(np.std(all_variances)),
        }
        mean_per_vector = np.mean(all_variances, axis=1)
        if num_database is not None and num_database < len(mean_per_vector):
            db_means = mean_per_vector[:num_database]
            q_means = mean_per_vector[num_database:]
            stats.update({
                "db_mean": float(np.mean(db_means)),
                "db_std": float(np.std(db_means)),
                "q_mean": float(np.mean(q_means)),
                "q_std": float(np.std(q_means)),
                "q_min": float(np.min(q_means)),
                "q_max": float(np.max(q_means)),
            })
        else:
            stats.update({
                "mean": float(np.mean(mean_per_vector)),
                "std": float(np.std(mean_per_vector)),
                "min": float(np.min(mean_per_vector)),
                "max": float(np.max(mean_per_vector)),
            })
    
    # Add scalar stats too for backward compatibility if needed
    stats.update({
        "all_min": float(np.min(all_variances)),
        "all_max": float(np.max(all_variances)),
        "all_median": float(np.median(all_variances)),
    })
    
    return stats


from eval_metrics.eval_ece_sh import compute_ece, compute_ece_pairwise

def compute_uncertainty_aucpr(
    predictions, positives_per_query, query_variances,
    uncertainty_loss="gaussian_nll",
    db_variances=None, q_desc=None, db_desc=None,
    n_values=[1, 5, 10], output_dir=None, distances=None,
    ece_metrics=["recall", "map"], zoom_threshold: float = 0.001, bin_mode: str = "zoom",
    vmf_kappa_floor: bool = False, percentile_two_sided: bool = False):
    """
    Computes AUC-PR for failure prediction using the model's uncertainty,
    following the same pattern as SUE's sue.py (lines 133-150):
      1. Binary labels: 1 if top-1 retrieval is correct, 0 otherwise.
      2. Confidence scores normalized to [0.1, 1.0] via np.interp.
      3. precision_recall_curve → auc.

    For vMF: kappa (concentration) is used directly as confidence (higher = more certain).
    For Gaussian: negative variance is used as confidence (lower variance = more certain).
    
    If db_variances and all_descriptors are provided (vMF only), also computes
    joint kappa AUC-PR:
      S      = sqrt(κ_Q² + κ_Ref² + 2·κ_Q·κ_Ref·(μ_Q^T·μ_Ref))
      S_norm = sqrt(κ_Q² + κ̂_Ref² + 2·κ_Q·κ̂_Ref·(μ_Q^T·μ_Ref))
      where κ̂_Ref = ln(κ_Ref + 1)
    
    Returns a dict: {'kappa': float, 'joint_kappa': float, 'joint_kappa_norm': float, ...}
    Includes ECE results if output_dir is provided.
    """
    logger.info("Computing AUC-PR for uncertainty (kappa)...")

    total_queries = len(predictions)

    # 1. Binary labels: 1 if top-1 is a ground-truth match
    matched = np.zeros(total_queries, dtype="float32")
    for i, preds in enumerate(predictions):
        if np.any(np.isin(preds[:1], positives_per_query[i])):
            matched[i] = 1.0

    # 2. Confidence scores from model uncertainty
    mean_q_var = np.mean(query_variances, axis=1)

    if uncertainty_loss.lower() == "vmf":
        # kappa is concentration: higher → more confident
        scores = mean_q_var
    else:
        # variance: lower → more confident, so negate
        scores = -1 * mean_q_var

    # Normalize to [0.0, 0.9999] (same range as SUE in sue.py)
    scores = np.interp(scores, (scores.min(), scores.max()), (0.0, 0.9999))

    # 3. Precision-recall curve and AUC
    precision, recall, _ = precision_recall_curve(matched, scores)
    aucpr_kappa = float(auc(recall, precision))
    results = {
        "kappa": aucpr_kappa,
        "kappa_min": float(np.min(mean_q_var)),
        "kappa_max": float(np.max(mean_q_var)),
        "kappa_mean": float(np.mean(mean_q_var)),
    }
    logger.info(f"AUC-PR (kappa):            {aucpr_kappa:.4f} [min: {results['kappa_min']:.4f}, max: {results['kappa_max']:.4f}, mean: {results['kappa_mean']:.4f}]")

    # 4. Joint kappa (vMF only) — resultant vector magnitude as confidence
    if (uncertainty_loss.lower() == "vmf" and db_variances is not None 
            and q_desc is not None and db_desc is not None):
        
        # κ_Q per query (mean across dimensions, since vMF outputs 1 scalar repeated)
        kappa_q = mean_q_var  # already computed above
        
        # κ_Ref for each top-K retrieved match
        mean_db_var = np.mean(db_variances, axis=1)  # per-database-image kappa
        K = predictions.shape[1]

        # Normalize to unit vectors for cosine similarity
        q_norms = np.linalg.norm(q_desc, axis=1, keepdims=True)
        db_norms = np.linalg.norm(db_desc, axis=1, keepdims=True)
        q_normed = q_desc / np.maximum(q_norms, 1e-8)
        db_normed = db_desc / np.maximum(db_norms, 1e-8)

        # Compute joint kappa S for all top-K pairs:
        #   S(q, r) = sqrt(κ_Q² + κ_Ref² + 2·κ_Q·κ_Ref·(μ_Q^T·μ_Ref))
        S_top_k = np.zeros((total_queries, K), dtype=np.float64)
        kappa_q_col = kappa_q[:, None]  # [NQ, 1] for broadcasting
        for k in range(K):
            ref_idx = predictions[:, k]
            kappa_ref_k = mean_db_var[ref_idx]
            mu_dot_k = np.sum(q_normed * db_normed[ref_idx], axis=1)
            S_top_k[:, k] = np.sqrt(np.maximum(
                kappa_q_col[:, 0] ** 2 + kappa_ref_k ** 2
                + 2 * kappa_q_col[:, 0] * kappa_ref_k * mu_dot_k,
                0.0,
            ))

        # Top-1 S values drive the existing AUC-PR and per-query joint-kappa ECE.
        S = S_top_k[:, 0]
        S_scores = np.interp(S, (S.min(), S.max()), (0.0, 0.9999))
        precision_j, recall_j, _ = precision_recall_curve(matched, S_scores)
        results["joint_kappa"] = float(auc(recall_j, precision_j))
        results["joint_kappa_min"] = float(np.min(S))
        results["joint_kappa_max"] = float(np.max(S))
        results["joint_kappa_mean"] = float(np.mean(S))
        logger.info(f"AUC-PR (joint kappa):      {results['joint_kappa']:.4f} [min: {results['joint_kappa_min']:.4f}, max: {results['joint_kappa_max']:.4f}, mean: {results['joint_kappa_mean']:.4f}]")

        # ECE for joint kappa (per-query, top-1 S values)
        ece_jk = compute_ece(
            predictions, positives_per_query, S[:, None],
            n_values=n_values, metrics=ece_metrics, output_dir=output_dir,
            distances=distances, uncertainty_loss="vmf", plot_filename="ece_joint_kappa.png",
            zoom_threshold=zoom_threshold, bin_mode=bin_mode,
            vmf_kappa_floor=vmf_kappa_floor,
            percentile_two_sided=percentile_two_sided,
        )
        results["ece_joint_kappa"] = ece_jk

        # Pairwise ECE on top-K joint-kappa S values (higher S = more certain; inverted in fn).
        ece_jk_pairwise = compute_ece_pairwise(
            predictions, positives_per_query, S_top_k,
            n_values=n_values, output_dir=output_dir,
            plot_filename="ece_pairwise_joint_kappa.png",
            zoom_threshold=zoom_threshold, bin_mode=bin_mode,
            uncertainty_loss="vmf", vmf_kappa_floor=vmf_kappa_floor,
            percentile_two_sided=percentile_two_sided,
        )
        results["ece_joint_kappa_pairwise"] = ece_jk_pairwise
        
        # # S_norm with κ̂_Ref = ln(κ_Ref + 1)
        # kappa_ref_norm = np.log(kappa_ref + 1)
        # S_norm = np.sqrt(np.maximum(
        #     kappa_q**2 + kappa_ref_norm**2 + 2 * kappa_q * kappa_ref_norm * mu_dot,
        #     0.0
        # ))
        # S_norm_scores = np.interp(S_norm, (S_norm.min(), S_norm.max()), (0.0, 0.9999))
        # precision_jn, recall_jn, _ = precision_recall_curve(matched, S_norm_scores)
        # results["joint_kappa_norm"] = float(auc(recall_jn, precision_jn))
        # logger.info(f"AUC-PR (joint kappa norm): {results['joint_kappa_norm']:.4f}")

        # # ECE for joint kappa norm
        # ece_jkn = compute_ece(
        #     predictions, positives_per_query, S_norm[:, None],
        #     n_values=n_values, metrics=ece_metrics, output_dir=output_dir,
        #     distances=distances, uncertainty_loss="vmf", plot_filename="ece_joint_kappa_norm.png",
        #     zoom_threshold=zoom_threshold,
        # )
        # results["ece_joint_kappa_norm"] = ece_jkn

        # # Scaled joint kappa norm ECE — normalize S_norm to [0, 0.9999]
        # S_norm_scaled = np.interp(S_norm, (S_norm.min(), S_norm.max()), (0.0, 0.9999))
        # ece_jkn_scaled = compute_ece(
        #     predictions, positives_per_query, S_norm_scaled[:, None],
        #     n_values=n_values, metrics=ece_metrics, output_dir=output_dir,
        #     distances=distances, uncertainty_loss="vmf", plot_filename="ece_joint_kappa_norm_scaled.png",
        #     zoom_threshold=zoom_threshold,
        # )
        # results["ece_joint_kappa_norm_scaled"] = ece_jkn_scaled

    results["raw_scores"] = {
        "kappa": mean_q_var,
        "joint_kappa": S if 'S' in locals() else None,
        "joint_kappa_pairwise": S_top_k if 'S_top_k' in locals() else None,
    }

    return results