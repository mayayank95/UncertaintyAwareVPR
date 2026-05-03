"""Baseline confidence scores (L2 / PA-score / SUE) for failure prediction in VPR.

All baselines operate on the same retrieval results (predictions, distances,
ground-truth) so they are directly comparable to the model's own uncertainty
(kappa / variance).

Implementation follows sue.py from the SUE paper exactly.

Baselines implemented:
  - L2 distance (match_scores)
  - PA-score (pa_scores)
  - SUE (sue_scores)
"""

import logging
import math

import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from eval_metrics.eval_ece_sh import compute_ece, compute_ece_pairwise

logger = logging.getLogger(__name__)


def compute_baselines(
    predictions,
    positives_per_query,
    distances,
    database_utms=None,
    n_values=None,
    output_dir=None,
    zoom_threshold: float = 0.001,
    bin_mode: str = "zoom",
):
    logger.debug(f"baselines: Computing for {len(predictions)} queries. output_dir={output_dir}")
    """
    Compute baseline confidence methods (AUC-PR + ECE + raw scores), exactly as in sue.py.

    Parameters
    ----------
    predictions : ndarray, shape [num_queries, K]
        Top-K retrieved indices (from FAISS).
    positives_per_query : list of arrays
        Ground-truth positive indices per query.
    distances : ndarray, shape [num_queries, K]
        Squared L2 distances from FAISS. We take sqrt to get L2.
    database_utms : ndarray, shape [num_database, 2]
        UTM coordinates of database images (needed for SUE).
    output_dir : Path or str, optional
        If provided, save ECE plots here.

    Returns
    -------
    dict  {baseline_name: scalar_or_dict}
        Includes AUC-PR values, ECE dictionaries, and raw scores per baseline.
    """
    total_queries = len(predictions)

    # --- FAISS returns squared L2; convert to L2 (same scale as sue.py's dists) ---
    dists = np.sqrt(np.maximum(distances, 0))

    # --- matched_array_foraucpr (sue.py lines 49-58) ---
    matched_array_foraucpr = np.zeros(total_queries, dtype="float32")
    l2distances = np.zeros(total_queries)

    for itr in range(total_queries): 
        if np.any(np.in1d(predictions[itr][:1], positives_per_query[itr])): #checking if Top-1 contains GT
            matched_array_foraucpr[itr] = 1.0
        l2distances[itr] = dists[itr][0]

    positive_rate = float(np.mean(matched_array_foraucpr))

    results = {}

    # ---- L2 distance (sue.py lines 73-74) ----
    match_scores = -1 * l2distances
    match_scores = np.interp(match_scores, (match_scores.min(), match_scores.max()), (0.0, 0.9999))

    precision_l2, recall_l2, _ = precision_recall_curve(matched_array_foraucpr, match_scores)
    results["l2"] = float(auc(recall_l2, precision_l2))
    results["l2_min"] = float(np.min(l2distances))
    results["l2_max"] = float(np.max(l2distances))
    results["l2_mean"] = float(np.mean(l2distances))
    logger.info(f"AUC-PR (L2 distance):  {results['l2']:.4f} [min: {results['l2_min']:.4f}, max: {results['l2_max']:.4f}, mean: {results['l2_mean']:.4f}]")

    if n_values:
        ece_l2 = compute_ece(
            predictions,
            positives_per_query,
            l2distances[:, None],
            n_values=n_values,
            output_dir=output_dir,
            plot_filename="ece_l2.png",
            zoom_threshold=zoom_threshold,
            bin_mode=bin_mode,
            percentile_two_sided=False,
        )
        results["ece_l2"] = ece_l2["ece_recall"]

        # Pairwise ECE on top-K L2 pair distances (higher distance = more uncertain).
        l2_pair_scores = dists  # [num_queries, K], already sqrt of squared L2 from FAISS.
        ece_l2_pairwise = compute_ece_pairwise(
            predictions,
            positives_per_query,
            l2_pair_scores,
            n_values=n_values,
            output_dir=output_dir,
            plot_filename="ece_pairwise_l2.png",
            zoom_threshold=zoom_threshold,
            bin_mode=bin_mode,
            uncertainty_loss="gaussian_nll",
            percentile_two_sided=False,
        )
        results["ece_l2_pairwise"] = ece_l2_pairwise

    # ---- PA-score (sue.py lines 76-81) ----
    if dists.shape[1] >= 2:
        pa_scores = np.zeros(total_queries)
        for itr in range(total_queries):
            pa_scores[itr] = dists[itr][0] / dists[itr][1]

        pa_scores_confidence = -1 * pa_scores
        pa_scores_normalized = np.interp(pa_scores_confidence, (pa_scores_confidence.min(), pa_scores_confidence.max()), (0.0, 0.9999))

        precision_pa, recall_pa, _ = precision_recall_curve(matched_array_foraucpr, pa_scores_normalized)
        results["pa"] = float(auc(recall_pa, precision_pa))
        results["pa_min"] = float(np.min(pa_scores))
        results["pa_max"] = float(np.max(pa_scores))
        results["pa_mean"] = float(np.mean(pa_scores))
        logger.info(f"AUC-PR (PA-score):     {results['pa']:.4f} [min: {results['pa_min']:.4f}, max: {results['pa_max']:.4f}, mean: {results['pa_mean']:.4f}]")

        if n_values:
            # Pass raw pa_scores (d1/d2), where higher = more uncertain
            ece_pa = compute_ece(
                predictions,
                positives_per_query,
                pa_scores[:, None],
                n_values=n_values,
                uncertainty_loss="gaussian_nll",
                output_dir=output_dir,
                plot_filename="ece_pa.png",
                zoom_threshold=zoom_threshold,
                bin_mode=bin_mode,
                percentile_two_sided=False,
            )
            results["ece_pa"] = ece_pa["ece_recall"]

    # ---- SUE (sue.py lines 95-131) ----
    if database_utms is not None:
        num_NN = 10
        slope = 350

        sue_scores = np.zeros(total_queries)
        weights = np.ones(num_NN)

        for itr in range(total_queries):
            top_preds = predictions[itr][:num_NN]
            nn_poses = database_utms[top_preds]
            bm_pose = nn_poses[0]

            for itr2 in range(num_NN):
                weights[itr2] = math.e ** ((-1 * abs(dists[itr][itr2])) * slope)

            weights = weights / sum(abs(weights))

            mean_pose = np.asarray([np.average(nn_poses[:, 0], weights=weights),
                                    np.average(nn_poses[:, 1], weights=weights)])

            variance_lat_lat = 0
            variance_lon_lon = 0
            variance_lat_lon = 0

            for k in range(0, num_NN):
                diff_lat_lat = min(500, nn_poses[k, 0] - mean_pose[0])
                diff_lon_lon = min(500, nn_poses[k, 1] - mean_pose[1])
                diff_lat_lon = min(500, nn_poses[k, 0] - mean_pose[0]) * min(500, nn_poses[k, 1] - mean_pose[1])

                variance_lat_lat = variance_lat_lat + weights[k] * (diff_lat_lat) ** 2
                variance_lon_lon = variance_lon_lon + weights[k] * (diff_lon_lon) ** 2
                variance_lat_lon = variance_lat_lon + weights[k] * diff_lat_lon

            sue_scores[itr] = (variance_lat_lat + variance_lon_lon) / 2

        sue_scores_confidence = -1 * sue_scores
        sue_scores_normalized = np.interp(sue_scores_confidence, (sue_scores_confidence.min(), sue_scores_confidence.max()), (0.0, 0.9999))

        precision_sue, recall_sue, _ = precision_recall_curve(matched_array_foraucpr, sue_scores_normalized)
        results["sue"] = float(auc(recall_sue, precision_sue))
        results["sue_min"] = float(np.min(sue_scores))
        results["sue_max"] = float(np.max(sue_scores))
        results["sue_mean"] = float(np.mean(sue_scores))
        logger.info(f"AUC-PR (SUE):          {results['sue']:.4f} [min: {results['sue_min']:.3f}, max: {results['sue_max']:.3f}, mean: {results['sue_mean']:.3f}]")

        if n_values:
            # Pass raw sue_scores (variance), where higher = more uncertain
            ece_sue = compute_ece(
                predictions,
                positives_per_query,
                sue_scores[:, None],
                n_values=n_values,
                uncertainty_loss="gaussian_nll",
                output_dir=output_dir,
                plot_filename="ece_sue.png",
                zoom_threshold=zoom_threshold,
                bin_mode=bin_mode,
                percentile_two_sided=False,
            )
            results["ece_sue"] = ece_sue["ece_recall"]

            # Log-transformed SUE scores (to handle wide range of spatial variances)
            sue_scores_log = np.log(sue_scores + 1e-10)
            results["sue_log_min"] = float(np.min(sue_scores_log))
            results["sue_log_max"] = float(np.max(sue_scores_log))
            results["sue_log_mean"] = float(np.mean(sue_scores_log))
            
            ece_sue_log = compute_ece(
                predictions,
                positives_per_query,
                sue_scores_log[:, None],
                n_values=n_values,
                uncertainty_loss="gaussian_nll",
                output_dir=output_dir,
                plot_filename="ece_sue_log.png",
                zoom_threshold=zoom_threshold,
                bin_mode=bin_mode,
                percentile_two_sided=False,
            )
            results["ece_sue_log"] = ece_sue_log["ece_recall"]
    elif n_values:
        logger.warning("No UTM coordinates found; skipping SUE ECE calculation.")

    results["random_baseline"] = positive_rate
    results["raw_scores"] = {
        "l2": l2distances,
        "pa": pa_scores if 'pa_scores' in locals() else None,
        "sue": sue_scores if 'sue_scores' in locals() else None,
        "sue_log": sue_scores_log if 'sue_scores_log' in locals() else None,
        "l2_pairwise": dists,
    }
    logger.info(f"AUC-PR baselines done  (positive rate = {positive_rate:.3f}, "
                f"random baseline ≈ {positive_rate:.3f})")

    return results
