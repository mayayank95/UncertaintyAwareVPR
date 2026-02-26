"""
Small demo to sanity-check the ECE calculation.

Run from project root:  python -m eval_metrics.demo_ece

Two scenarios:
  A) Well-calibrated: low variance -> high recall, high variance -> low recall  => ECE low
  B) Poorly calibrated: variance uncorrelated with recall                        => ECE high
"""

import logging
import sys
from pathlib import Path

import numpy as np

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_metrics.eval_ece_sh import compute_ece

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def make_synthetic_data(
    num_queries: int,
    num_db: int,
    max_k: int,
    dim: int,
    well_calibrated: bool,
    seed: int = 42,
):
    """Create predictions, positives, and query variances.

    well_calibrated=True:  positive rank is gradual in variance.
      Low var  -> rank 0-1 (R@1=100%, R@5=100%, R@10=100%)
      Mid var  -> rank 2-5 (R@1=0%,  R@5=100%, R@10=100%)
      High var -> rank 6-10 or outside (R@1=0%, R@5=0%, R@10=100% or 0%)
    well_calibrated=False: random assignment (variance and recall unrelated).
    """
    rng = np.random.default_rng(seed)

    # Variances: [num_queries, dim], spread in [0.05, 0.95]
    query_variances = rng.uniform(0.05, 0.95, (num_queries, dim)).astype(np.float32)

    # Mean variance per query (what we bin on)
    mean_var = np.mean(query_variances, axis=1)

    # For each query, one positive DB index (ground truth)
    positives_per_query = [rng.integers(0, num_db, size=1) for _ in range(num_queries)]

    # Predictions: [num_queries, max_k] — predicted DB indices in order of similarity
    predictions = np.zeros((num_queries, max_k), dtype=np.int64)

    for q in range(num_queries):
        pos = positives_per_query[q][0]
        others = np.array([i for i in range(num_db) if i != pos])
        rng.shuffle(others)

        if well_calibrated:
            # Map mean_var in [0,1] to rank in 0..max_k (or max_k+1 = outside)
            # Low var -> rank 0; high var -> rank max_k or outside
            rank = int(np.clip(mean_var[q] * (max_k + 1), 0, max_k + 1))
            if rank <= max_k:
                predictions[q, :rank] = others[:rank]
                predictions[q, rank] = pos
                predictions[q, rank + 1 :] = others[rank : max_k - 1]
            else:
                predictions[q, :] = others[:max_k]
        else:
            predictions[q, :] = rng.permutation(num_db)[:max_k]

        positives_per_query[q] = np.array([pos])

    return predictions, positives_per_query, query_variances


def main():
    num_queries = 200
    num_db = 500
    max_k = 10
    dim = 32
    n_values = [1, 5, 10]
    num_bins = 11

    print("=" * 60)
    print("ECE Demo — synthetic retrieval data")
    print("=" * 60)

    # ---- Scenario A: Well-calibrated ----
    print("\n--- Scenario A: Well-calibrated (low var -> high recall) ---")
    pred_a, pos_a, var_a = make_synthetic_data(
        num_queries, num_db, max_k, dim, well_calibrated=True, seed=1
    )
    result_a = compute_ece(pred_a, pos_a, var_a, n_values=n_values, num_bins=num_bins)
    print(f"ECE_R@1/5/10: {result_a['ece_recall'][1]:.3f} / {result_a['ece_recall'][5]:.3f} / {result_a['ece_recall'][10]:.3f}")
    print(f"ECE_mAP@1/5/10: {result_a['ece_map'][1]:.3f} / {result_a['ece_map'][5]:.3f} / {result_a['ece_map'][10]:.3f}")
    print("(Expected: low ECE when uncertainty matches performance)")

    # ---- Scenario B: Poorly calibrated ----
    print("\n--- Scenario B: Poorly calibrated (random) ---")
    pred_b, pos_b, var_b = make_synthetic_data(
        num_queries, num_db, max_k, dim, well_calibrated=False, seed=2
    )
    result_b = compute_ece(pred_b, pos_b, var_b, n_values=n_values, num_bins=num_bins)
    print(f"ECE_R@1/5/10: {result_b['ece_recall'][1]:.3f} / {result_b['ece_recall'][5]:.3f} / {result_b['ece_recall'][10]:.3f}")
    print(f"ECE_mAP@1/5/10: {result_b['ece_map'][1]:.3f} / {result_b['ece_map'][5]:.3f} / {result_b['ece_map'][10]:.3f}")
    print("(Expected: higher ECE when variance and recall are unrelated)")

    print("\n" + "=" * 60)
    print("Demo done. Scenario A should have lower ECE than Scenario B.")
    print("=" * 60)


if __name__ == "__main__":
    main()
