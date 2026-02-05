import logging
import sys

import json
from pathlib import Path

import faiss
import numpy as np
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Local module imports
from configs.parser import build_config, init_model
from data.test_dataset import TestDataset
from data.upload_dataset import upload_dataset
from losses.cosface_loss import cosine_distance
from utils import visualizations

# Initialize Logger
logger = logging.getLogger(__name__)

def _compute_correlation(distances, variances):
    """Compute Pearson correlation between distances and variances."""
    if len(distances) > 1 and np.std(distances) > 0 and np.std(variances) > 0:
        corr, _ = pearsonr(distances, variances)
        return corr
    return 0.0

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
    )
    logger.info(f"{'='*30}\nTesting on {test_ds}")

    # --- 1. Combined Descriptor & Variance Extraction ---
    # We store both to avoid re-running the model for uncertainty calculations
    all_descriptors = np.empty((len(test_ds), args['descriptors_dimension']), dtype="float32")
    all_variances = np.empty((len(test_ds), args['descriptors_dimension']), dtype="float32")

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

    # --- 3. Uncertainty Correlation (Optimized) ---
    uncertainty_corr = 0.0
    if args['model_mode'] == "uncertainty" and args['use_labels'] and not args['dry_run'] and args['datasets_type'] == ['test']:
        logger.info("Computing uncertainty correlation metrics...")
        loss_type = args.get('uncertainty_loss', 'gaussian_nll').lower()
        
        # Get query indices and their first positive ground truth from DB
        # Filter out queries with no positives
        valid_queries = [(i, pos[0]) for i, pos in enumerate(positives_per_query) if len(pos) > 0]
        
        if len(valid_queries) > 0:
            q_indices = np.array([i for i, _ in valid_queries])
            db_gt_indices = np.array([idx for _, idx in valid_queries])
            
            # Convert to tensors for normalization/distance math
            q_tensor = torch.from_numpy(all_descriptors[test_ds.num_database + q_indices])
            db_tensor = torch.from_numpy(all_descriptors[db_gt_indices])
            q_var_tensor = torch.from_numpy(all_variances[test_ds.num_database + q_indices])

            q_norm = torch.nn.functional.normalize(q_tensor, p=2, dim=1)
            db_norm = torch.nn.functional.normalize(db_tensor, p=2, dim=1)

            if loss_type == 'gaussian_cosine':
                dists = cosine_distance(q_norm, db_norm)
            else:
                dists = torch.sum((q_norm - db_norm) ** 2, dim=-1)      
            mean_vars = torch.mean(q_var_tensor, dim=-1)
            uncertainty_corr = _compute_correlation(dists.numpy(), mean_vars.numpy())

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
        logger.info(
            f"Uncertainty Pearson Correlation: {uncertainty_corr:.4f}, "
            f"Variance (Mean: {np.mean(all_variances):.4e}, Min: {np.min(all_variances):.4e}, Max: {np.max(all_variances):.4e})"
        )

    return recalls, recalls_str, uncertainty_corr

if __name__ == "__main__":
    cfg, entries = build_config()

    if cfg.get("resume_model"):
        old_log_dir = Path(cfg['log_dir'])
        train_dir = Path(cfg['resume_model']).parent
        new_log_dir = train_dir / "eval"
        new_log_dir.mkdir(parents=True, exist_ok=True)
        cfg['log_dir'] = str(new_log_dir)

        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                root_logger.removeHandler(handler)
                file_name = Path(handler.baseFilename).name
                new_handler = logging.FileHandler(new_log_dir / file_name, encoding="utf-8")
                
                # Move existing log file to new directory
                old_file = old_log_dir / file_name
                new_file = new_log_dir / file_name
                if old_file.exists():
                    old_file.rename(new_file)

                new_handler = logging.FileHandler(new_file, encoding="utf-8")
                new_handler.setFormatter(handler.formatter)
                new_handler.setLevel(handler.level)
                root_logger.addHandler(new_handler)
        
        logger.info(f"Saving evaluation results to: {cfg['log_dir']}")

        if cfg.get("save_config"):
            old_config = old_log_dir / "merged_config.json"
            new_config = new_log_dir / "merged_config.json"
            if old_config.exists():
                old_config.rename(new_config)
                logger.info(f"Moved merged config to {new_config}")
            else:
                new_config.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                logger.info(f"Saved merged config to {new_config}")

        # Cleanup empty timestamp folder
        try:
            old_log_dir.rmdir()
        except OSError:
            pass

    datasets_paths = upload_dataset(cfg, entries)
    device, model = init_model(cfg)

    for entry in entries:
        name = entry['name']
        logger.info(f"Starting evaluation: {name}")
        
        recalls, r_str, corr = eval_dataset(
            cfg, model, device, name, datasets_paths[name]['test']
        )

    logger.info("="*30 + "\nAll processes finished.")