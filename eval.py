import logging
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from configs.runtime import build_config_and_datasets, init_model, init_wandb, log_wandb_images
from data.test_dataset import TestDataset
from eval_metrics import visualizations
from eval_metrics.eval_ece_sh import compute_ece, _cal_mapk
from eval_metrics.uncertainty import compute_uncertainty_correlation, compute_uncertainty_statistics
from utils import commons, wandb_utils

logger = logging.getLogger(__name__)



def eval_dataset(args, model, device, dataset_name, eval_ds_path, wandb_step=None, log_dataset_info=True):
    """
    Evaluates the model on a single dataset.
    Extracts features once, then computes Recalls and Uncertainty metrics.
    log_dataset_info: if True, log dataset size (e.g. "Testing on ..."); set False when called every epoch from train.
    """
    model = model.eval()
    # Use eval/<dataset_name> so we never conflict with a file named like the dataset (e.g. sf_xl)
    dataset_output_dir = Path(args['log_dir']) / "eval" / dataset_name
    # When using W&B, create directories even in dry_run mode so logging has a valid target.
    if not args['dry_run'] or args.get("use_wandb"):
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

    test_ds = TestDataset(
        f"{eval_ds_path}/database",
        f"{eval_ds_path}/queries",
        positive_dist_threshold=args['positive_dist_threshold'],
        image_size=args.get('image_size'),
        use_labels=args['use_labels'],
        resize_test_imgs=args.get('resize_test_imgs', False),
    )
    if log_dataset_info:
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
        logger.debug("Extracting database features...")
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
        logger.debug("Extracting query features...")
        for images, indices in tqdm(q_loader):
            desc, var = model(images.to(device))
            all_descriptors[indices.numpy(), :] = desc.cpu().numpy()
            all_variances[indices.numpy(), :] = var.cpu().numpy()
            if args["dry_run"]: break

    # Split for FAISS and evaluation
    db_desc = all_descriptors[:test_ds.num_database]
    q_desc = all_descriptors[test_ds.num_database:]
    
    if args['dry_run']:
        db_desc = db_desc[:max(args['infer_batch_size'], args.get('num_preds_to_save', 10))]
        q_desc = q_desc[:max(1, args.get('num_queries_to_save', 3))]

    if args.get('save_descriptors') and not args['dry_run'] and args['datasets_type'] == ['test']:
        np.save(dataset_output_dir / "queries_descriptors.npy", q_desc)
        np.save(dataset_output_dir / "database_descriptors.npy", db_desc)

    # --- 2. Similarity Search & Recalls ---
    faiss_index = faiss.IndexFlatL2(args['descriptors_dimension'])
    faiss_index.add(db_desc)
    distances, predictions = faiss_index.search(q_desc, max(args['recall_values']))

    recalls_str = "Labels not available"
    recalls = np.zeros(len(args['recall_values']))
    positives_per_query = None

    map_at_k = None
    if args['use_labels']:
        positives_per_query = test_ds.get_positives()
        for query_idx, preds in enumerate(predictions):
            for i, n in enumerate(args['recall_values']):
                if np.any(np.isin(preds[:n], positives_per_query[query_idx])):
                    recalls[i:] += 1
                    break
        recalls = recalls / test_ds.num_queries * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args['recall_values'], recalls)])
        map_at_k = [_cal_mapk(predictions, positives_per_query, n) for n in args['recall_values']]
        
        if not args['dry_run'] and args['datasets_type'] == ['test']:
            (dataset_output_dir / "recalls.txt").write_text(recalls_str)



    # --- 3. Uncertainty Metrics ---
    uncertainty_corr = None
    mean_query_variance = None
    std_query_variance = None
    min_query_variance = None
    max_query_variance = None
    ece_result = None
    if args['model_mode'] == "uncertainty":
        q_var = all_variances[test_ds.num_database:]
        if len(q_var) > 0:
            mean_per_q = np.mean(q_var, axis=1)
            mean_query_variance = float(np.mean(mean_per_q))
            std_query_variance = float(np.std(mean_per_q))
            min_query_variance = float(np.min(mean_per_q))
            max_query_variance = float(np.max(mean_per_q))
        save_corr_plot = args.get('save_plots', False) or args.get('use_wandb')
        if save_corr_plot:
            dataset_output_dir.mkdir(parents=True, exist_ok=True)
        uncertainty_corr = compute_uncertainty_correlation(
            args, all_descriptors, all_variances, positives_per_query, test_ds.num_database,
            output_dir=dataset_output_dir if save_corr_plot else None,
        )
        if args['use_labels'] and positives_per_query is not None:
            q_variances = all_variances[test_ds.num_database:]
            if args['dry_run']:
                q_variances = q_variances[:max(1, args.get('num_queries_to_save', 3))]
            # Respect --ece_metrics only (default from parser is recall-only).
            ece_metrics = args.get("ece_metrics")
            if not ece_metrics:
                ece_metrics = ["recall"]
            save_ece_plot = not args['dry_run'] or args.get('use_wandb')
            if save_ece_plot:
                dataset_output_dir.mkdir(parents=True, exist_ok=True)
            ece_result = compute_ece(
                predictions, positives_per_query, q_variances,
                n_values=args['recall_values'],
                output_dir=dataset_output_dir if save_ece_plot else None,
                metrics=ece_metrics,
                distances=distances,
                uncertainty_loss=args.get('uncertainty_loss', 'gaussian_nll'),
            )
    # --- 4. Visualizations & Plots ---
    save_plots = args.get("save_plots", False) or args.get("use_wandb", False)

    if save_plots and args.get('num_preds_to_save', 0) != 0:
        preds_to_save = predictions[:, :args['num_preds_to_save']]
        pred_distances = distances[:, :args['num_preds_to_save']]
        query_variances = np.mean(all_variances[test_ds.num_database:], axis=1)
        db_var = all_variances[:test_ds.num_database] if args['model_mode'] == 'uncertainty' else None
        visualizations.save_preds(
            preds_to_save, test_ds, str(dataset_output_dir),
            args['save_only_wrong_preds'], args['use_labels'], args['num_queries_to_save'],
            distances=pred_distances, query_variances=query_variances, db_variances=db_var,
            all_descriptors=all_descriptors,
        )

    if args['use_labels']:
        msg = f"Results for {dataset_name}: {recalls_str}"
        if map_at_k is not None:
            msg += " | " + ", ".join([f"mAP@{val}: {m:.2f}" for val, m in zip(args['recall_values'], map_at_k)])
        logger.info(msg)
        if recalls[0] == 0 and map_at_k is not None and map_at_k[0] == 0:
            logger.info("R@1 and mAP@1 are 0 when no query has a positive at rank 1 (e.g. dry run with random descriptors, or frozen backbone with poor retrieval).")
        # With frozen backbone, R@k and mAP@k above are constant across epochs (same descriptors). Only ECE (bin-based) changes as variance changes.
    # Save variance distribution (for val section in W&B when training, or when save_plots)
    save_var_plot = save_plots or args.get('use_wandb')
    if args['model_mode'] == "uncertainty" and save_var_plot:
        if dataset_output_dir.exists() and not dataset_output_dir.is_dir():
            logger.warning("Output path %s exists as a file; skipping variance distribution plot.", dataset_output_dir)
        else:
            dataset_output_dir.mkdir(parents=True, exist_ok=True)
            if uncertainty_corr is not None and save_plots:
                logger.info(f"Uncertainty Spearman Correlation: {uncertainty_corr:.4f}")
            compute_uncertainty_statistics(
                all_variances,
                dataset_output_dir,
                num_database=test_ds.num_database,
            )

    # Collect all W&B metrics for the caller to log.
    # Prefixed keys (Eval_{dataset_name}/...) for grouping per dataset in eval runs.
    # Flat keys (val/loss total, ece/recall_01, ...) when wandb_step is set so train.py time-series graphs get data.
    prefix = f"Eval_{dataset_name}/"
    wandb_metrics = {}
    if ece_result:
        if "ece_recall" in ece_result:
            for n, v in ece_result["ece_recall"].items():
                k = f"{n:02d}"  # recall_01, recall_05 so all ece recalls group together
                wandb_metrics[f"{prefix}ece_recall_{k}"] = float(v)
                if wandb_step is not None:
                    wandb_metrics[f"ece/recall_{k}"] = float(v)
        if "ece_map" in ece_result:
            for n, v in ece_result["ece_map"].items():
                k = f"{n:02d}"
                wandb_metrics[f"{prefix}ece_map_{k}"] = float(v)
                if wandb_step is not None:
                    wandb_metrics[f"ece/map_{k}"] = float(v)
        if "ece_ap" in ece_result:
            wandb_metrics[f"{prefix}ece_ap"] = float(ece_result["ece_ap"])
            if wandb_step is not None:
                wandb_metrics["ece/ap"] = float(ece_result["ece_ap"])

    # Build image dict for W&B when plots exist (train: val/ece sections; eval: always for diagnostic plots)
    wandb_images = None
    if args.get("use_wandb") and dataset_output_dir.exists():
        wandb_images = {
            f"{prefix}variance_distribution": (dataset_output_dir / "variance_distribution.png").resolve(),
            f"{prefix}ece_plot": (dataset_output_dir / "ece_plot.png").resolve(),
            f"{prefix}uncertainty_correlation_scatter": (dataset_output_dir / "uncertainty_correlation_scatter.png").resolve(),
        }
        preds_dir = dataset_output_dir / "preds"
        if save_plots and preds_dir.exists():
            wandb_images[f"{prefix}predictions"] = []
            for p in sorted(preds_dir.glob("*.jpg")):
                wandb_images[f"{prefix}predictions"].append(p)

    return recalls, recalls_str, map_at_k, uncertainty_corr, mean_query_variance, std_query_variance, min_query_variance, max_query_variance, wandb_metrics, wandb_images

if __name__ == "__main__":
    # ---- Load config and datasets (shared helper) ----
    cfg, entries, datasets_paths = build_config_and_datasets()
    init_wandb(cfg, job_type="eval")

    device, model = init_model(cfg)

    # Optionally copy the resume model into the current log directory
    commons.copy_resume_model_to_log_dir(cfg, logger)

    for entry in entries:
        name = entry["name"]
        logger.info(f"Starting evaluation: {name}")

        eval_ds_path = datasets_paths[name]["test"]
        if not eval_ds_path.exists():
            eval_ds_path = datasets_paths[name]["validation"]
            logger.info(f"[{name}] 'test' folder not found, using 'val' instead.")
        recalls, r_str, map_at_k, corr, mean_var, std_var, min_var, max_var, eval_wb, wandb_images = eval_dataset(
            cfg, model, device, name, eval_ds_path, wandb_step=None
        )

        wandb_utils.log_eval_dataset(
            cfg, name, recalls, map_at_k, corr, mean_var, std_var, min_var, max_var, eval_wb, wandb_images
        )

        # Populate W&B Summary for this dataset with the key scalar metrics.
        if cfg.get("use_wandb"):
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    prefix = f"Eval_{name}"
                    rv = cfg.get("recall_values", [1, 5, 10, 20])
                    for k in sorted(rv):
                        i = rv.index(k)
                        _wandb.run.summary[f"{prefix}/recall_{k:02d}"] = float(recalls[i]) if i < len(recalls) else 0.0
                        if map_at_k is not None and i < len(map_at_k):
                            _wandb.run.summary[f"{prefix}/map_{k:02d}"] = float(map_at_k[i])
                        elif map_at_k is not None:
                            _wandb.run.summary[f"{prefix}/map_{k:02d}"] = 0.0

                    # ECE values (already computed inside eval_dataset)
                    for k in sorted(rv):
                        k2 = f"{k:02d}"
                        ecr_key = f"{prefix}/ece_recall_{k2}"
                        if ecr_key in eval_wb:
                            _wandb.run.summary[ecr_key] = float(eval_wb[ecr_key])
                        ecm_key = f"{prefix}/ece_map_{k2}"
                        if ecm_key in eval_wb:
                            _wandb.run.summary[ecm_key] = float(eval_wb[ecm_key])

                    eap_key = f"{prefix}/ece_ap"
                    if eap_key in eval_wb:
                        _wandb.run.summary[eap_key] = float(eval_wb[eap_key])

                    # Variance statistics (passed separately from eval_dataset)
                    if corr is not None:
                        _wandb.run.summary[f"{prefix}/uncertainty_correlation"] = float(corr)
                    if mean_var is not None:
                        _wandb.run.summary[f"{prefix}/variance_mean"] = float(mean_var)
                    if std_var is not None:
                        _wandb.run.summary[f"{prefix}/variance_std"] = float(std_var)
                    if min_var is not None:
                        _wandb.run.summary[f"{prefix}/variance_min"] = float(min_var)
                    if max_var is not None:
                        _wandb.run.summary[f"{prefix}/variance_max"] = float(max_var)
            except Exception:
                # Best-effort only; don't fail evaluation due to summary writing.
                pass

    logger.info("=" * 30 + "\nAll processes finished.")
    wandb_utils.finish_run(cfg)