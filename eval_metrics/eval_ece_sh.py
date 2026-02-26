"""Expected Calibration Error (ECE) for uncertainty-aware VPR.

Bins queries by predicted uncertainty (mean variance), computes recall@N and
mAP@N per bin, and measures how well uncertainty tracks retrieval performance.

A well-calibrated model should have high recall in low-uncertainty bins and low
recall in high-uncertainty bins, with a smooth monotonic relationship.

ECE = Σ (bin_weight * |bin_metric - expected_metric|)  over all bins.

Adapted from: https://github.com/ramdrop/stun
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_zoomed_bins(variances: np.ndarray, num_bins: int):
    """Equal-width bins with adaptive zoom to handle long-tailed distributions.

    Iteratively narrows the upper range until the last bin has at least 0.1%
    of the data, preventing empty high-uncertainty bins from outliers.

    Returns:
        indices: list of arrays, each containing query indices for that bin.
        k: number of zoom steps applied.
    """
    s_min, s_max = np.min(variances), np.max(variances)
    bins_parent = np.linspace(s_min, s_max, num=num_bins)
    k = 0
    while True:
        indices = []
        bins_child = np.linspace(bins_parent[0], bins_parent[-1 - k], num=num_bins)
        for i in range(num_bins - 1):
            left = np.where(variances >= bins_child[i])
            if i != num_bins - 2:
                right = np.where(variances < bins_child[i + 1])
            else:
                right = np.where(variances <= bins_child[i + 1])
            indices.append(np.intersect1d(left[0], right[0]))
        if len(indices[-1]) > int(variances.shape[0] * 0.001) or k == num_bins - 2:
            break
        k += 1
    return indices, k


def _cal_recall(predictions: np.ndarray, positives_per_query: List, n_values: List[int]) -> np.ndarray:
    """Compute recall@N. Returns array of shape [len(n_values)], values in [0, 100]."""
    recalls = np.zeros(len(n_values))
    num_queries = predictions.shape[0]
    if num_queries == 0:
        return recalls
    for q_idx in range(num_queries):
        for i, n in enumerate(n_values):
            if np.sum(np.in1d(predictions[q_idx, :n], positives_per_query[q_idx])) > 0:
                recalls[i:] += 1
                break
    return recalls / num_queries * 100.0


def _cal_apk(positives, prediction, k):
    """Average precision at K for a single query."""
    if len(prediction) > k:
        prediction = prediction[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(prediction):
        if p in positives and p not in prediction[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(positives), k) * 100.0


def _cal_mapk(predictions, positives_per_query, k):
    """Mean average precision at K across all queries."""
    return np.mean([_cal_apk(pos, pred, k) for pos, pred in zip(positives_per_query, predictions)])


def compute_ece(
    predictions: np.ndarray,
    positives_per_query: List,
    query_variances: np.ndarray,
    n_values: List[int] = [1, 5, 10],
    num_bins: int = 11,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Compute Expected Calibration Error for uncertainty-aware retrieval.

    Args:
        predictions: [num_queries, max_k] predicted DB indices per query.
        positives_per_query: list of arrays, ground-truth positive DB indices per query.
        query_variances: [num_queries, D] variance vectors for each query.
        n_values: recall@N values to evaluate.
        num_bins: number of bin edges (actual bins = num_bins - 1).
        output_dir: if provided, save ECE plot here.

    Returns:
        dict with 'ece_recall', 'ece_map', 'bin_recalls', 'bin_map', 'bin_weights'.
    """
    mean_var = np.mean(query_variances, axis=-1)
    bin_indices, zoom_k = _get_zoomed_bins(mean_var, num_bins)
    num_actual_bins = num_bins - 1
    num_queries = len(mean_var)

    bin_recalls = np.zeros((num_actual_bins, len(n_values)))
    bin_map = np.zeros((num_actual_bins, len(n_values)))
    bin_weights = np.zeros(num_actual_bins)

    ece_recall = np.zeros(len(n_values))
    ece_map = np.zeros(len(n_values))

    for b, q_in_bin in enumerate(bin_indices):
        if len(q_in_bin) == 0:
            continue
        bin_weights[b] = len(q_in_bin) / num_queries

        bin_preds = predictions[q_in_bin]
        bin_positives = [positives_per_query[i] for i in q_in_bin]

        # Expected performance: linearly decreasing from 1.0 (low uncertainty) to 0.0 (high uncertainty)
        expected = (num_actual_bins - 1 - b) / (num_actual_bins - 1)

        # Recall@N
        recall_at_n = _cal_recall(bin_preds, bin_positives, n_values)
        bin_recalls[b] = recall_at_n
        for i in range(len(n_values)):
            ece_recall[i] += bin_weights[b] * abs(recall_at_n[i] / 100.0 - expected)

        # mAP@N
        map_at_n = [_cal_mapk(bin_preds, bin_positives, n) for n in n_values]
        bin_map[b] = map_at_n
        for i in range(len(n_values)):
            ece_map[i] += bin_weights[b] * abs(map_at_n[i] / 100.0 - expected)

    # Log results
    ece_rec_str = "/".join([f"{e:.3f}" for e in ece_recall])
    ece_map_str = "/".join([f"{e:.3f}" for e in ece_map])
    n_str = "/".join([str(n) for n in n_values])
    logger.info(f"ECE_R@{n_str}: {ece_rec_str}  (zoom_k={zoom_k})")
    logger.info(f"ECE_mAP@{n_str}: {ece_map_str}")

    if output_dir:
        _plot_ece(bin_recalls, bin_map, bin_weights, bin_indices, n_values, num_actual_bins, output_dir)

    return {
        "ece_recall": {n: e for n, e in zip(n_values, ece_recall)},
        "ece_map": {n: e for n, e in zip(n_values, ece_map)},
        "bin_recalls": bin_recalls,
        "bin_map": bin_map,
        "bin_weights": bin_weights,
    }


def _plot_ece(bin_recalls, bin_map, bin_weights, bin_indices, n_values, num_bins, output_dir):
    """Save ECE visualization plot."""
    try:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Bin distribution
        ax = axs[0][0]
        ax.bar(range(num_bins), [len(x) for x in bin_indices])
        ax.set_xlabel("σ² (uncertainty: low → high)")
        ax.set_ylabel("Number of samples")

        # Recall per bin
        ax = axs[0][1]
        x = np.arange(num_bins)
        for i, n in enumerate(n_values):
            ax.plot(x, bin_recalls[:, i], marker="o", label=f"R@{n}")
        ax.set_xlabel("σ² (uncertainty: low → high)")
        ax.set_ylabel("Recall@N")
        ax.legend()

        # mAP per bin
        ax = axs[1][0]
        for i, n in enumerate(n_values):
            ax.plot(x, bin_map[:, i], marker="o", label=f"mAP@{n}")
        ax.set_xlabel("σ² (uncertainty: low → high)")
        ax.set_ylabel("mAP@N")
        ax.legend()

        # Weights
        ax = axs[1][1]
        ax.bar(range(num_bins), bin_weights)
        ax.set_xlabel("σ² (uncertainty: low → high)")
        ax.set_ylabel("Bin weight (fraction of queries)")

        plt.tight_layout()
        plt.savefig(Path(output_dir) / "ece_plot.png", dpi=150)
        plt.close()
        logger.info(f"ECE plot saved to {output_dir}/ece_plot.png")
    except ImportError:
        logger.warning("matplotlib not installed, skipping ECE plot.")
