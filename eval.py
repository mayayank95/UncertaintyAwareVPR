import logging
from pathlib import Path

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from configs.runtime import build_config_and_datasets, init_model
from data.test_dataset import TestDataset
from eval_metrics import visualizations
from eval_metrics.eval_ece_sh import compute_ece
from eval_metrics.uncertainty import compute_uncertainty_correlation, compute_uncertainty_statistics, normalize_variance
from utils import commons

logger = logging.getLogger(__name__)

def eval_dataset(args, model, device, dataset_name, eval_ds_path):
    """
    Evaluates the model on a single dataset.
    Extracts features once, then computes Recalls and Uncertainty metrics.
    """
    model = model.eval()
    dataset_output_dir = Path(args['log_dir']) / dataset_name
    if not args['dry_run']:
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

    test_ds = TestDataset(
        f"{eval_ds_path}/database",
        f"{eval_ds_path}/queries",
        positive_dist_threshold=args['positive_dist_threshold'],
        image_size=args.get('image_size'),
        use_labels=args['use_labels'],
        resize_test_imgs=args.get('resize_test_imgs', False),
    )
    logger.info(f"{'='*30}\nTesting on {test_ds}")

    # --- 1. Combined Descriptor & Variance Extraction ---
    # We store both to avoid re-running the model for uncertainty calculations
    all_descriptors = np.zeros((len(test_ds), args['descriptors_dimension']), dtype="float32")
    all_variances = np.zeros((len(test_ds), args['descriptors_dimension']), dtype="float32")

    if args['dry_run']:
        logger.info("Dry run enabled: Generating random descriptors/variances to test correlation pipeline.")
        all_descriptors = np.random.randn(len(test_ds), args['descriptors_dimension']).astype("float32")
        all_variances = np.abs(np.random.randn(len(test_ds), args['descriptors_dimension']).astype("float32"))

    with torch.inference_mode():
        # Database extraction
        db_subset = Subset(test_ds, list(range(test_ds.num_database)))
        db_loader = DataLoader(
            db_subset, batch_size=args['infer_batch_size'], 
            num_workers=args['num_workers'], pin_memory=(device.type == "cuda")
        )
        logger.info("Extracting database features...")
        for images, indices in tqdm(db_loader):
            desc, var = model(images.to(device))
            all_descriptors[indices.numpy(), :] = desc.cpu().numpy()
            all_variances[indices.numpy(), :] = var.cpu().numpy()
            if args["dry_run"]: break

        # Query extraction
        q_subset = Subset(test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries)))
        q_loader = DataLoader(
            q_subset, batch_size=1, 
            num_workers=args['num_workers'], pin_memory=(device.type == "cuda")
        )
        logger.info("Extracting query features...")
        for images, indices in tqdm(q_loader):
            desc, var = model(images.to(device))
            all_descriptors[indices.numpy(), :] = desc.cpu().numpy()
            all_variances[indices.numpy(), :] = var.cpu().numpy()
            if args["dry_run"]: break

    # Split for FAISS and evaluation
    db_desc = all_descriptors[:test_ds.num_database]
    q_desc = all_descriptors[test_ds.num_database:]
    
    if args['dry_run']:
        db_desc = db_desc[:args['infer_batch_size']]
        q_desc = q_desc[:1]

    if args.get('save_descriptors') and not args['dry_run'] and args['datasets_type'] == ['test']:
        np.save(dataset_output_dir / "queries_descriptors.npy", q_desc)
        np.save(dataset_output_dir / "database_descriptors.npy", db_desc)

    # --- 2. Similarity Search & Recalls ---
    faiss_index = faiss.IndexFlatL2(args['descriptors_dimension'])
    faiss_index.add(db_desc)
    _, predictions = faiss_index.search(q_desc, max(args['recall_values']))

    recalls_str = "Labels not available"
    recalls = np.zeros(len(args['recall_values']))
    positives_per_query = None

    if args['use_labels']:
        positives_per_query = test_ds.get_positives()
        for query_idx, preds in enumerate(predictions):
            for i, n in enumerate(args['recall_values']):
                if np.any(np.isin(preds[:n], positives_per_query[query_idx])):
                    recalls[i:] += 1
                    break
        recalls = recalls / test_ds.num_queries * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args['recall_values'], recalls)])
        
        if not args['dry_run'] and args['datasets_type'] == ['test']:
            (dataset_output_dir / "recalls.txt").write_text(recalls_str)

    # --- 3. Uncertainty Metrics ---
    uncertainty_corr = None
    if args['model_mode'] == "uncertainty":
        uncertainty_corr = compute_uncertainty_correlation(
            args, all_descriptors, all_variances, positives_per_query, test_ds.num_database
        )
        if args['use_labels'] and positives_per_query is not None:
            q_variances = all_variances[test_ds.num_database:]
            if args['dry_run']:
                q_variances = q_variances[:1]
            if args.get('normalize_variance'):
                q_variances = normalize_variance(q_variances, args['normalize_variance'])
            compute_ece(
                predictions, positives_per_query, q_variances,
                n_values=args['recall_values'],
                output_dir=dataset_output_dir if not args['dry_run'] else None,
            )
    # --- 4. Visualizations ---
    if args.get('num_preds_to_save', 0) != 0 and not args['dry_run'] and args['datasets_type'] == ['test']:
        visualizations.save_preds(
            predictions[:, :args['num_preds_to_save']], 
            test_ds, str(dataset_output_dir), 
            args['save_only_wrong_preds'], args['use_labels'], args['num_queries_to_save']
        )

    if args['datasets_type'] == ['test']:
        logger.info(f"Results for {dataset_name}: {recalls_str}")
    if args['model_mode'] == "uncertainty" and args['datasets_type'] == ['test']:
        if uncertainty_corr is not None:
            logger.info(f"Uncertainty Pearson Correlation: {uncertainty_corr:.4f}")
        compute_uncertainty_statistics(all_variances, dataset_output_dir if not args['dry_run'] else None)

    return recalls, recalls_str, uncertainty_corr

if __name__ == "__main__":
    # ---- Load config and datasets (shared helper) ----
    cfg, entries, datasets_paths = build_config_and_datasets()

    device, model = init_model(cfg)

    # Optionally copy the resume model into the current log directory
    commons.copy_resume_model_to_log_dir(cfg, logger)

    for entry in entries:
        name = entry["name"]
        logger.info(f"Starting evaluation: {name}")

        recalls, r_str, corr = eval_dataset(
            cfg, model, device, name, datasets_paths[name]["test"]
        )

    logger.info("=" * 30 + "\nAll processes finished.")