import logging
from pathlib import Path
from typing import Dict

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc

from configs.runtime import build_config_and_datasets, init_model, init_wandb, log_wandb_images
from data.test_dataset import TestDataset
from eval_metrics import visualizations
from eval_metrics.eval_ece_sh import (
    compute_ece,
    compute_ece_pairwise,
    percentile_two_sided_from_var_head,
    _cal_mapk,
)
# from eval_metrics.pairwise_scores import get_all_pairwise_scores
from eval_metrics.uncertainty import (compute_uncertainty_statistics,
                                     compute_uncertainty_aucpr)
from eval_metrics.baselines import compute_baselines
from utils import commons, wandb_utils

logger = logging.getLogger(__name__)



def eval_dataset(args, model, device, dataset_name, eval_ds_path, queries_folder_name="queries", 
                 wandb_step=None, log_dataset_info=True, db_features=None):
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
        f"{eval_ds_path}/{queries_folder_name}",
        positive_dist_threshold=args['positive_dist_threshold'],
        image_size=args.get('image_size'),
        use_labels=args['use_labels'],
        resize_test_imgs=args.get('resize_test_imgs', False),
    )
    if log_dataset_info:
        logger.info(f"{'='*30}\nTesting on {test_ds}")

    # --- 1. Combined Descriptor & Variance Extraction ---
    # Check if saved descriptors/variances exist on disk (from a previous --save_descriptors run)
    q_desc_path = dataset_output_dir / "queries_descriptors.npy"
    db_desc_path = dataset_output_dir / "database_descriptors.npy"
    q_var_path = dataset_output_dir / "queries_variances.npy"
    db_var_path = dataset_output_dir / "database_variances.npy"

    loaded_from_disk = False
    if q_desc_path.exists() and db_desc_path.exists():
        db_desc = np.load(db_desc_path)
        q_desc = np.load(q_desc_path)
        if len(db_desc) == test_ds.num_database and len(q_desc) == test_ds.num_queries:
            logger.info(f"Loading saved descriptors from {dataset_output_dir}")
            all_descriptors = np.vstack([db_desc, q_desc])
            
            if q_var_path.exists() and db_var_path.exists():
                db_var_loaded = np.load(db_var_path)
                q_var_loaded = np.load(q_var_path)
                if len(db_var_loaded) == test_ds.num_database and len(q_var_loaded) == test_ds.num_queries:
                    all_variances = np.vstack([db_var_loaded, q_var_loaded])
                    logger.info(f"Loaded saved variances from {dataset_output_dir}")
                else:
                    all_variances = np.zeros_like(all_descriptors)
            else:
                all_variances = np.zeros_like(all_descriptors)
            
            loaded_from_disk = True
        else:
            logger.warning(f"Saved descriptors in {dataset_output_dir} size mismatch "
                           f"(DB: {len(db_desc)} vs {test_ds.num_database}, Q: {len(q_desc)} vs {test_ds.num_queries}); "
                           "ignoring disk cache and re-extracting.")
    
    if not loaded_from_disk:
        # We store both to avoid re-running the model for uncertainty calculations
        all_descriptors = np.zeros((len(test_ds), args['descriptors_dimension']), dtype="float32")
        all_variances = np.zeros((len(test_ds), args['descriptors_dimension']), dtype="float32")

        db_cache_found = False
        if db_features is not None:
            logger.info("Using in-memory cached database features.")
            all_descriptors[:test_ds.num_database] = db_features['descriptors']
            all_variances[:test_ds.num_database] = db_features['variances']
            db_cache_found = True

        with torch.inference_mode():
            # Database extraction (skip if in-memory cache found)
            if not db_cache_found:
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
                
                # Prepare for sharing with subsequent query folders in the same run
                db_features = {
                    'descriptors': all_descriptors[:test_ds.num_database].copy(),
                    'variances': all_variances[:test_ds.num_database].copy()
                }

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
            np.save(db_desc_path, db_desc)
            np.save(q_desc_path, q_desc)
            if args['model_mode'] == 'uncertainty':
                q_var = all_variances[test_ds.num_database:]
                db_var = all_variances[:test_ds.num_database]
                np.save(q_var_path, q_var)
                np.save(db_var_path, db_var)
                logger.info(f"Saved descriptors and variances to {dataset_output_dir}")


    # --- 2. Similarity Search & Recalls ---
    faiss_index = faiss.IndexFlatL2(args['descriptors_dimension'])
    faiss_index.add(db_desc)
    max_recall_k = max(args['recall_values'])
    search_k = min(max_recall_k, db_desc.shape[0])
    distances, predictions = faiss_index.search(q_desc, search_k)

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
        map_at_k = None
        if not args.get('only_recalls'):
            map_at_k = [_cal_mapk(predictions[:, :max_recall_k], positives_per_query, n) for n in args['recall_values']]
        
        if not args['dry_run'] and args['datasets_type'] == ['test']:
            (dataset_output_dir / "recalls.txt").write_text(recalls_str)

    # Trim predictions/distances to max_recall_k for downstream blocks
    # (ECE, AUC-PR, visualizations all assume [NQ, max_recall_k]).
    if predictions.shape[1] > max_recall_k:
        predictions = predictions[:, :max_recall_k]
        distances = distances[:, :max_recall_k]



    # --- 3. Uncertainty Metrics ---
    uncertainty_corr = None
    mean_query_variance = None
    std_query_variance = None
    min_query_variance = None
    max_query_variance = None
    ece_result = None
    uncertainty_aucpr = None
    baseline_results = None
    uncertainty_stats = None
    if args['model_mode'] == "uncertainty":
        q_var = all_variances[test_ds.num_database:]
        if len(q_var) > 0:
            mean_per_q = np.mean(q_var, axis=1)
            mean_query_variance = float(np.mean(mean_per_q))
            std_query_variance = float(np.std(mean_per_q))
            min_query_variance = float(np.min(mean_per_q))
            max_query_variance = float(np.max(mean_per_q))
        # save_corr_plot = args.get('save_plots', False) or args.get('use_wandb')
        # if save_corr_plot:
        #     dataset_output_dir.mkdir(parents=True, exist_ok=True)
        # uncertainty_corr = compute_uncertainty_correlation(
        #     args, q_desc, db_desc, q_variances, positives_per_query,
        #     output_dir=dataset_output_dir if save_corr_plot else None,
        # )
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
                zoom_threshold=args.get('ece_zoom_threshold', 0.001),
                bin_mode=args.get('ece_bin_mode', 'zoom'),
                vmf_kappa_floor=args.get('ece_vmf_kappa_floor', False),
                percentile_two_sided=percentile_two_sided_from_var_head(args.get("var_head_type")),
            )
                       
            # --- Detailed metrics for standalone Evaluation only (skip during training or if flag set) ---
            logger.debug(f"Checking for detailed metrics: wandb_step={wandb_step}, skip_detailed={args.get('skip_detailed_metrics')}")
            if wandb_step is None and not args.get('skip_detailed_metrics'):
                # --- AUC-PR for failure prediction ---
                logger.debug("Computing detailed uncertainty AUC-PR and baselines...")
                db_var = all_variances[:test_ds.num_database]
                uncertainty_aucpr = compute_uncertainty_aucpr(
                    predictions, positives_per_query, q_variances,
                    uncertainty_loss=args.get('uncertainty_loss', 'gaussian_nll'),
                    db_variances=db_var,
                    q_desc=q_desc,
                    db_desc=db_desc,
                    n_values=args.get("recall_values"),
                    output_dir=dataset_output_dir if save_ece_plot else None,
                    distances=distances,
                    ece_metrics=ece_metrics,
                    zoom_threshold=args.get('ece_zoom_threshold', 0.001),
                    bin_mode=args.get('ece_bin_mode', 'zoom'),
                    vmf_kappa_floor=args.get('ece_vmf_kappa_floor', False),
                    percentile_two_sided=percentile_two_sided_from_var_head(args.get("var_head_type")),
                )
                
                # --- Baselines (L2, PA-score, SUE) for comparison ---
                db_utms = getattr(test_ds, 'database_utms', None)
                baseline_results = compute_baselines(
                    predictions, positives_per_query, distances,
                    database_utms=db_utms,
                    n_values=args.get("recall_values"),
                    output_dir=dataset_output_dir if save_ece_plot else None,
                    zoom_threshold=args.get('ece_zoom_threshold', 0.001),
                    bin_mode=args.get('ece_bin_mode', 'zoom'),
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
            q_desc=q_desc, db_desc=db_desc,
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
            uncertainty_stats = compute_uncertainty_statistics(
                all_variances,
                dataset_output_dir,
                num_database=test_ds.num_database,
            )

    # Collect all W&B metrics for the caller to log.
    # Prefixed keys (Eval_{dataset_name}/...) for grouping per dataset in eval runs.
    # Flat keys (val/loss total, ece/recall_01, ...) when wandb_step is set so train.py time-series graphs get data.
    prefix = f"Eval_{dataset_name}/"
    wandb_metrics = {}
    rv = args.get("recall_values", [1, 5, 10, 20])
    for k in sorted(rv):
        i = rv.index(k)
        if i < len(recalls):
            wandb_metrics[f"{prefix}recall_{k:02d}"] = float(recalls[i])
            if map_at_k is not None and i < len(map_at_k):
                wandb_metrics[f"{prefix}map_{k:02d}"] = float(map_at_k[i])
    
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
                        
    if uncertainty_aucpr:
        for name, val in uncertainty_aucpr.items():
            if name == "raw_scores":
                continue
            if isinstance(val, dict):
                # ECE dictionary (e.g., ece_joint_kappa: {ece_recall: {1: 0.1, ...}, ...})
                for metric_name, metric_vals in val.items():
                    if metric_name in ["ece_recall", "ece_map"]:
                        for n, e_val in metric_vals.items():
                            k_name = f"{prefix}{name}_{metric_name}_{n:02d}"
                            wandb_metrics[k_name] = float(e_val)
                            if wandb_step is not None:
                                wandb_metrics[f"uncertainty/{name}_{metric_name}_{n:02d}"] = float(e_val)
                    elif metric_name == "ece_ap":
                        wandb_metrics[f"{prefix}{name}_{metric_name}"] = float(metric_vals)
                        if wandb_step is not None:
                            wandb_metrics[f"uncertainty/{name}_{metric_name}"] = float(metric_vals)
            else:
                if any(suffix in name for suffix in ["_min", "_max", "_mean"]):
                    m_prefix = "stats_"
                else:
                    m_prefix = "auc_pr_"
                wandb_metrics[f"{prefix}{m_prefix}{name}"] = float(val)
                if wandb_step is not None:
                    wandb_metrics[f"uncertainty/{m_prefix}{name}"] = float(val)
    
    if baseline_results:
        for name, val in baseline_results.items():
            if name == "raw_scores":
                continue
            if isinstance(val, dict):
                # val is an ECE dict like {'ece_recall': {1: 0.1, ...}, 'ece_map': {...}}
                for metric_type in ["ece_recall", "ece_map"]:
                    if metric_type in val:
                        for n, e_val in val[metric_type].items():
                            wandb_metrics[f"{prefix}baseline_{name}_{metric_type}_{n:02d}"] = float(e_val)
                            if wandb_step is not None:
                                wandb_metrics[f"uncertainty/baseline_{name}_{metric_type}_{n:02d}"] = float(e_val)
            else:
                if any(suffix in name for suffix in ["_min", "_max", "_mean"]):
                    m_prefix = "stats_"
                else:
                    m_prefix = "auc_pr_"
                wandb_metrics[f"{prefix}{m_prefix}{name}"] = float(val)
                if wandb_step is not None:
                    wandb_metrics[f"uncertainty/{m_prefix}{name}"] = float(val)

    if uncertainty_stats:
        for name, val in uncertainty_stats.items():
            wandb_metrics[f"{prefix}uncertainty_{name}"] = float(val)
            if wandb_step is not None:
                wandb_metrics[f"uncertainty/stats_{name}"] = float(val)

    # Build image dict for W&B when plots exist (train: val/ece sections; eval: always for diagnostic plots)
    wandb_images = {}
    if args.get("use_wandb") and dataset_output_dir.exists():
        expected_plots = {
            # Variance distribution
            "variance_distribution": "variance_distribution.png",
            # ECE kappa (main model uncertainty)
            "ece_plot": "ece_plot.png",
            "ece_plot_bins": "ece_plot_bins.png",
            "ece_plot_weights": "ece_plot_weights.png",
            # ECE joint kappa
            "ece_joint_kappa": "ece_joint_kappa.png",
            "ece_joint_kappa_bins": "ece_joint_kappa_bins.png",
            "ece_joint_kappa_weights": "ece_joint_kappa_weights.png",
            # ECE L2
            "ece_l2": "ece_l2.png",
            "ece_l2_bins": "ece_l2_bins.png",
            "ece_l2_weights": "ece_l2_weights.png",
            # ECE PA-score
            "ece_pa": "ece_pa.png",
            "ece_pa_bins": "ece_pa_bins.png",
            "ece_pa_weights": "ece_pa_weights.png",
            # ECE SUE
            "ece_sue": "ece_sue.png",
            "ece_sue_bins": "ece_sue_bins.png",
            "ece_sue_weights": "ece_sue_weights.png",
            # ECE Log-SUE
            "ece_sue_log": "ece_sue_log.png",
            "ece_sue_log_bins": "ece_sue_log_bins.png",
            "ece_sue_log_weights": "ece_sue_log_weights.png",
            # Pairwise ECE L2
            "ece_pairwise_l2": "ece_pairwise_l2.png",
            "ece_pairwise_l2_bins": "ece_pairwise_l2_bins.png",
            "ece_pairwise_l2_weights": "ece_pairwise_l2_weights.png",
            # Pairwise ECE joint kappa
            "ece_pairwise_joint_kappa": "ece_pairwise_joint_kappa.png",
            "ece_pairwise_joint_kappa_bins": "ece_pairwise_joint_kappa_bins.png",
            "ece_pairwise_joint_kappa_weights": "ece_pairwise_joint_kappa_weights.png",
        }
        for name, filename in expected_plots.items():
            plot_path = dataset_output_dir / filename
            if plot_path.exists():
                wandb_images[f"{prefix}{name}"] = plot_path.resolve()
            # else:
            #     logger.debug(f"Optional plot {filename} not found, skipping W&B log.")

        # Per–Top-K pairwise distribution plots (ece_*_bins_rN.png / ece_*_weights_rN.png, combined_* variants).
        for pattern in ("*pairwise*_bins_r*.png", "*pairwise*_weights_r*.png"):
            for extra in sorted(dataset_output_dir.glob(pattern)):
                wandb_images[f"{prefix}{extra.stem}"] = extra.resolve()

        # Split-panel ECE: optional ``{stem}_map.png``, ``{stem}_ap.png``, ``{stem}_recall.png`` (when not primary).
        for pattern in ("ece_*_map.png", "ece_*_ap.png", "ece_*_recall.png"):
            for extra in sorted(dataset_output_dir.glob(pattern)):
                key = f"{prefix}{extra.stem}"
                if key not in wandb_images:
                    wandb_images[key] = extra.resolve()

        preds_dir = dataset_output_dir / "preds"
        if save_plots and preds_dir.exists():
            image_list = sorted(preds_dir.glob("*.jpg"))
            if image_list:
                wandb_images[f"{prefix}predictions"] = image_list

    if not wandb_images:
        wandb_images = None

    # Build aggregation data for combined ECE across datasets
    aggregation_data = None
    if (args['model_mode'] == "uncertainty" and args['use_labels'] 
            and positives_per_query is not None and not args['dry_run']):
        q_variances_agg = all_variances[test_ds.num_database:]
        aggregation_data = {
            "name": dataset_name,
            "predictions": predictions,
            "positives_per_query": positives_per_query,
            "q_variances": q_variances_agg,
            "distances": distances,
            "baseline_raw": baseline_results.get("raw_scores") if baseline_results else None,
            "uncertainty_raw": uncertainty_aucpr.get("raw_scores") if uncertainty_aucpr else None,
        }

    return recalls, recalls_str, map_at_k, uncertainty_corr, mean_query_variance, std_query_variance, min_query_variance, max_query_variance, wandb_metrics, wandb_images, db_features, aggregation_data

if __name__ == "__main__":
    # ---- Load config and datasets (shared helper) ----
    cfg, entries, datasets_paths = build_config_and_datasets()
    commons.make_deterministic(cfg["seed"])
    init_wandb(cfg, job_type="eval")

    device, model = init_model(cfg)

    # Optionally copy the resume model into the current log directory
    commons.copy_resume_model_to_log_dir(cfg, logger)

    all_agg_data = []  # Collect per-dataset data for combined ECE
    for entry in entries:
        name = entry["name"]
        logger.info(f"Starting evaluation: \033[1m{name}\033[0m")

        eval_ds_path = datasets_paths[name]["test"]
        if not eval_ds_path.exists():
            eval_ds_path = datasets_paths[name]["validation"]
            logger.info(f"[{name}] 'test' folder not found, using 'val' instead.")

        # Look for all folders starting with "queries" or "query"
        query_folders = sorted([p for p in eval_ds_path.glob("queries*") if p.is_dir()])
        if not query_folders:
            query_folders = sorted([p for p in eval_ds_path.glob("query*") if p.is_dir()])
        
        if not query_folders:
            logger.warning(f"[{name}] No query folders found in {eval_ds_path}")
            continue
        shared_db_features = None  # Reset for each test dataset
        for q_folder in query_folders:
            q_folder_name = q_folder.name
            # If there's only one folder and it's named "queries" or "query", use base name
            # Otherwise append the folder name to distinguish them in logs/W&B
            if len(query_folders) == 1 and q_folder_name in ["queries", "query"]:
                display_name = name
            else:
                display_name = f"{name}_{q_folder_name}"
            
            logger.info(f"Evaluating \033[1m{name}\033[0m - queries folder: {q_folder_name} (as \033[1m{display_name}\033[0m)")

            results = eval_dataset(
                cfg, model, device, display_name, eval_ds_path, queries_folder_name=q_folder_name, 
                wandb_step=None, db_features=shared_db_features
            )
            recalls, r_str, map_at_k, corr, mean_var, std_var, min_var, max_var, eval_wb, wandb_images, shared_db_features, agg_data = results
            if agg_data is not None:
                agg_data["base_dataset_name"] = name
                agg_data["query_folder_name"] = q_folder_name
                all_agg_data.append(agg_data)

            wandb_utils.log_eval_dataset(
                cfg, display_name, recalls, map_at_k, corr, mean_var, std_var, min_var, max_var, eval_wb, wandb_images
            )

            # Populate W&B Summary for this dataset with the key scalar metrics.
            if cfg.get("use_wandb"):
                try:
                    import wandb as _wandb
                    if _wandb.run is not None:
                        # 1. Log all scalar metrics returned by eval_dataset (includes recalls, mAP, ECE)
                        for k, v in eval_wb.items():
                            _wandb.run.summary[k] = v
                        
                        # 2. Log additional evaluation statistics
                        prefix = f"Eval_{display_name}"
                        # if corr is not None:
                        #     _wandb.run.summary[f"{prefix}/uncertainty_correlation"] = float(corr)
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

    def _compute_and_log_combined(agg_data_list, output_subdir, plot_prefix, wandb_prefix):
        combined_variances = np.vstack([d["q_variances"] for d in agg_data_list])
        combined_distances = np.vstack([d["distances"] for d in agg_data_list])

        combined_predictions_list = []
        combined_positives = []
        combined_matched = []
        db_offset = 0
        for d in agg_data_list:
            preds = d["predictions"]
            positives = d["positives_per_query"]
            max_db_idx = int(preds.max()) + 1 if len(preds) > 0 else 0

            combined_predictions_list.append(preds + db_offset)
            for i, pos_list in enumerate(positives):
                combined_positives.append(np.array(pos_list) + db_offset)
                is_hit = np.any(np.isin(preds[i, :1], pos_list))
                combined_matched.append(float(is_hit))

            db_offset += max_db_idx

        combined_predictions = np.vstack(combined_predictions_list)
        combined_matched = np.array(combined_matched)

        combined_output_dir = Path(cfg['log_dir']) / "eval" / output_subdir
        combined_output_dir.mkdir(parents=True, exist_ok=True)

        ece_metrics = cfg.get("ece_metrics") or ["recall"]
        uncertainty_loss = cfg.get('uncertainty_loss', 'gaussian_nll')

        combined_ece = compute_ece(
            combined_predictions, combined_positives, combined_variances,
            n_values=cfg['recall_values'], output_dir=combined_output_dir, metrics=ece_metrics,
            distances=combined_distances, uncertainty_loss=uncertainty_loss,
            zoom_threshold=cfg.get('ece_zoom_threshold', 0.001), plot_filename=f"{plot_prefix}.png",
            bin_mode=cfg.get('ece_bin_mode', 'zoom'),
            vmf_kappa_floor=cfg.get('ece_vmf_kappa_floor', False),
            percentile_two_sided=percentile_two_sided_from_var_head(cfg.get("var_head_type")),
        )

        combined_summary = {"ece_kappa": combined_ece}

        # --- Combined recall (original L2 order) ---
        total_q = len(combined_positives)
        if total_q > 0:
            orig_rec = np.zeros(len(cfg['recall_values']))
            for qi, preds in enumerate(combined_predictions):
                for i, n in enumerate(cfg['recall_values']):
                    if np.any(np.isin(preds[:n], combined_positives[qi])):
                        orig_rec[i:] += 1
                        break
            orig_rec = orig_rec / total_q * 100
            for k_val, v in zip(cfg['recall_values'], orig_rec):
                combined_summary[f"recall_{k_val:02d}"] = float(v)

        combined_mean_var = np.mean(combined_variances, axis=-1)
        kappa_scores = combined_mean_var if uncertainty_loss.lower() == "vmf" else -combined_mean_var
        kappa_norm = np.interp(kappa_scores, (kappa_scores.min(), kappa_scores.max()), (0.0, 0.9999))
        p, r, _ = precision_recall_curve(combined_matched, kappa_norm)
        combined_summary["auc_pr_kappa"] = float(auc(r, p))

        for b_name in ["l2", "pa", "sue", "sue_log"]:
            # For SUE variants, exclude amstertime from combined baseline calculation.
            baseline_agg_data = agg_data_list
            if b_name in {"sue", "sue_log"}:
                baseline_agg_data = [
                    d for d in agg_data_list
                    if "amstertime" not in d.get("base_dataset_name", "").lower()
                ]
                if len(baseline_agg_data) != len(agg_data_list):
                    logger.info(
                        f"{output_subdir}: excluding amstertime from combined {b_name} calculation "
                        f"({len(agg_data_list) - len(baseline_agg_data)} entries removed)."
                    )

            if baseline_agg_data and all(
                d["baseline_raw"] and b_name in d["baseline_raw"] and d["baseline_raw"][b_name] is not None
                for d in baseline_agg_data
            ):
                # Build baseline-specific combined structures (must match baseline score length).
                base_predictions_list = []
                base_positives = []
                base_matched = []
                base_distances = []
                base_db_offset = 0
                for d in baseline_agg_data:
                    preds = d["predictions"]
                    positives = d["positives_per_query"]
                    dists = d["distances"]
                    max_db_idx = int(preds.max()) + 1 if len(preds) > 0 else 0

                    base_predictions_list.append(preds + base_db_offset)
                    base_distances.append(dists)
                    for i, pos_list in enumerate(positives):
                        base_positives.append(np.array(pos_list) + base_db_offset)
                        is_hit = np.any(np.isin(preds[i, :1], pos_list))
                        base_matched.append(float(is_hit))
                    base_db_offset += max_db_idx

                base_predictions = np.vstack(base_predictions_list)
                base_distances = np.vstack(base_distances)
                base_matched = np.array(base_matched)

                b_scores = np.concatenate([d["baseline_raw"][b_name] for d in baseline_agg_data])
                b_ece = compute_ece(
                    base_predictions, base_positives, b_scores[:, None],
                    n_values=cfg['recall_values'], output_dir=combined_output_dir, metrics=ece_metrics,
                    distances=base_distances, uncertainty_loss="gaussian_nll",
                    zoom_threshold=cfg.get('ece_zoom_threshold', 0.001), plot_filename=f"{plot_prefix}_{b_name}.png",
                    bin_mode=cfg.get('ece_bin_mode', 'zoom'),
                    percentile_two_sided=False,
                )
                combined_summary[f"ece_{b_name}"] = b_ece

                b_conf = -b_scores
                b_norm = np.interp(b_conf, (b_conf.min(), b_conf.max()), (0.0, 0.9999))
                p_b, r_b, _ = precision_recall_curve(base_matched, b_norm)
                combined_summary[f"auc_pr_{b_name}"] = float(auc(r_b, p_b))

        # Pairwise ECE on L2 top-K distances (no amstertime exclusion — follows regular L2 baseline).
        if all(
            d["baseline_raw"] and d["baseline_raw"].get("l2_pairwise") is not None
            for d in agg_data_list
        ):
            l2_pair_scores_combined = np.vstack([
                d["baseline_raw"]["l2_pairwise"] for d in agg_data_list
            ])
            l2_pw_ece = compute_ece_pairwise(
                combined_predictions, combined_positives, l2_pair_scores_combined,
                n_values=cfg['recall_values'],
                output_dir=combined_output_dir,
                plot_filename=f"{plot_prefix}_pairwise_l2.png",
                zoom_threshold=cfg.get('ece_zoom_threshold', 0.001),
                bin_mode=cfg.get('ece_bin_mode', 'zoom'),
                uncertainty_loss="gaussian_nll",
                percentile_two_sided=False,
            )
            combined_summary["ece_l2_pairwise"] = l2_pw_ece

        if all(d["uncertainty_raw"] and "joint_kappa" in d["uncertainty_raw"] and d["uncertainty_raw"]["joint_kappa"] is not None for d in agg_data_list):
            jk_scores = np.concatenate([d["uncertainty_raw"]["joint_kappa"] for d in agg_data_list])
            jk_ece = compute_ece(
                combined_predictions, combined_positives, jk_scores[:, None],
                n_values=cfg['recall_values'], output_dir=combined_output_dir, metrics=ece_metrics,
                distances=combined_distances, uncertainty_loss="vmf",
                zoom_threshold=cfg.get('ece_zoom_threshold', 0.001), plot_filename=f"{plot_prefix}_joint_kappa.png",
                bin_mode=cfg.get('ece_bin_mode', 'zoom'),
                vmf_kappa_floor=cfg.get('ece_vmf_kappa_floor', False),
                percentile_two_sided=percentile_two_sided_from_var_head(cfg.get("var_head_type")),
            )
            combined_summary["ece_joint_kappa"] = jk_ece

            jk_norm = np.interp(jk_scores, (jk_scores.min(), jk_scores.max()), (0.0, 0.9999))
            p_jk, r_jk, _ = precision_recall_curve(combined_matched, jk_norm)
            combined_summary["auc_pr_joint_kappa"] = float(auc(r_jk, p_jk))

        # Pairwise ECE on joint-kappa top-K S values.
        if all(
            d["uncertainty_raw"] and d["uncertainty_raw"].get("joint_kappa_pairwise") is not None
            for d in agg_data_list
        ):
            jk_pair_scores_combined = np.vstack([
                d["uncertainty_raw"]["joint_kappa_pairwise"] for d in agg_data_list
            ])
            jk_pw_ece = compute_ece_pairwise(
                combined_predictions, combined_positives, jk_pair_scores_combined,
                n_values=cfg['recall_values'],
                output_dir=combined_output_dir,
                plot_filename=f"{plot_prefix}_pairwise_joint_kappa.png",
                zoom_threshold=cfg.get('ece_zoom_threshold', 0.001),
                bin_mode=cfg.get('ece_bin_mode', 'zoom'),
                uncertainty_loss="vmf",
                vmf_kappa_floor=cfg.get('ece_vmf_kappa_floor', False),
                percentile_two_sided=percentile_two_sided_from_var_head(cfg.get("var_head_type")),
            )
            combined_summary["ece_joint_kappa_pairwise"] = jk_pw_ece

        try:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2, figsize=(14, 5))
            ax = axs[0]
            for d in agg_data_list:
                ds_mean_var = np.mean(d["q_variances"], axis=-1)
                ds_label = d.get("name", "unknown")
                ax.hist(ds_mean_var, bins=50, alpha=0.5, label=f"{ds_label} (n={len(ds_mean_var)})", edgecolor="black", linewidth=0.3)
            ax.set_xlabel("Mean kappa / variance per query")
            ax.set_ylabel("Frequency")
            ax.set_title("Per-dataset query uncertainty distribution")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            ax = axs[1]
            ax.hist(combined_mean_var, bins=50, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.3)
            ax.axvline(np.median(combined_mean_var), color="tomato", linestyle="--", linewidth=1.5,
                       label=f"Median = {np.median(combined_mean_var):.4f}")
            ax.axvline(np.mean(combined_mean_var), color="orange", linestyle="-", linewidth=1.5,
                       label=f"Mean = {np.mean(combined_mean_var):.4f}")
            ax.set_xlabel("Mean kappa / variance per query")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Combined query uncertainty (n={len(combined_mean_var)})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            dist_path = combined_output_dir / f"{output_subdir}_kappa_distribution.png"
            plt.savefig(dist_path, dpi=150)
            plt.close(fig)
            logger.info(f"Combined kappa distribution plot saved to {dist_path}")
        except ImportError:
            logger.warning("matplotlib not installed, skipping combined kappa distribution plot.")
        except Exception as e:
            logger.warning(f"Error saving combined kappa distribution plot: {e}")

        if cfg.get("use_wandb"):
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    for m_name, m_val in combined_summary.items():
                        if isinstance(m_val, dict):
                            for metric_type in ["ece_recall", "ece_map"]:
                                if metric_type in m_val:
                                    for n, v in m_val[metric_type].items():
                                        _wandb.run.summary[f"{wandb_prefix}/{m_name}_{metric_type}_{n:02d}"] = float(v)
                        else:
                            _wandb.run.summary[f"{wandb_prefix}/{m_name}"] = float(m_val)

                    for fname in combined_output_dir.glob(f"{plot_prefix}*.png"):
                        key = fname.stem.replace(plot_prefix, "ece_plot")
                        if key == "ece_plot":
                            key = "ece_plot_kappa"
                        if key == "ece_plot_scaled":
                            continue
                        _wandb.log({f"{wandb_prefix}/{key}": _wandb.Image(str(fname))})

                    dist_path = combined_output_dir / f"{output_subdir}_kappa_distribution.png"
                    if dist_path.exists():
                        _wandb.log({f"{wandb_prefix}/kappa_distribution": _wandb.Image(str(dist_path))})
            except Exception as e:
                logger.warning(f"Failed to log {output_subdir} ECE to W&B: {e}")

        total_queries = len(combined_positives)
        n_datasets = len(agg_data_list)
        logger.info(f"{output_subdir}: combined ECE computed over {total_queries} queries from {n_datasets} datasets.")

    # --- Combined ECE variants ---
    if len(all_agg_data) > 1 and cfg['model_mode'] == 'uncertainty':
        # 1. sfxl-v1, sfxl-v2, msls-val-query, pitts30, amstertime
        combined_5 = [
            d for d in all_agg_data
            if (
                (("sf_xl" in d.get("base_dataset_name", "").lower() or "sf-xl" in d.get("base_dataset_name", "").lower())
                 and ("v1" in d.get("query_folder_name", "").lower() or "v2" in d.get("query_folder_name", "").lower())) or
                "amstertime" in d.get("base_dataset_name", "").lower() or
                "pitts30" in d.get("base_dataset_name", "").lower() or
                ("msls" in d.get("base_dataset_name", "").lower() and d.get("query_folder_name") == "query")
            )
        ]
        if len(combined_5) > 1:
            logger.info("=" * 30 + "\nComputing combined_5 (sfxl-v1/2, amstertime, pitts30, msls-query)...")
            _compute_and_log_combined(combined_5, "combined_5", "ece_combined_5", "Eval_combined_5")

        # 2. sfxl-v1, sfxl-v2, msls-val-query, pitts30
        combined_4 = [
            d for d in all_agg_data
            if (
                (("sf_xl" in d.get("base_dataset_name", "").lower() or "sf-xl" in d.get("base_dataset_name", "").lower())
                 and ("v1" in d.get("query_folder_name", "").lower() or "v2" in d.get("query_folder_name", "").lower())) or
                "pitts30" in d.get("base_dataset_name", "").lower() or
                ("msls" in d.get("base_dataset_name", "").lower() and d.get("query_folder_name") == "query")
            )
        ]
        if len(combined_4) > 1:
            logger.info("=" * 30 + "\nComputing combined_4 (sfxl-v1/2, pitts30, msls-query)...")
            _compute_and_log_combined(combined_4, "combined_4", "ece_combined_4", "Eval_combined_4")

        # 3. sfxl-v1, sfxl-v2, sf-xl-night, sf-occlusion, msls-val-query, pitts30
        combined_sf_all = [
            d for d in all_agg_data
            if (
                ("sf_xl" in d.get("base_dataset_name", "").lower() or "sf-xl" in d.get("base_dataset_name", "").lower()) or
                "pitts30" in d.get("base_dataset_name", "").lower() or
                ("msls" in d.get("base_dataset_name", "").lower() and d.get("query_folder_name") == "query")
            ) and "amstertime" not in d.get("base_dataset_name", "").lower()
        ]
        if len(combined_sf_all) > 1:
            logger.info("=" * 30 + "\nComputing combined_sf_all (sfxl all, pitts30, msls-query)...")
            _compute_and_log_combined(combined_sf_all, "combined_sf_all", "ece_combined_sf_all", "Eval_combined_sf_all")

        # 4. sfxl-all, msls-val-query, pitts30, amstertime
        combined_all = [
            d for d in all_agg_data
            if (
                ("sf_xl" in d.get("base_dataset_name", "").lower() or "sf-xl" in d.get("base_dataset_name", "").lower()) or
                "pitts30" in d.get("base_dataset_name", "").lower() or
                ("msls" in d.get("base_dataset_name", "").lower() and d.get("query_folder_name") == "query") or
                "amstertime" in d.get("base_dataset_name", "").lower()
            )
        ]
        if len(combined_all) > 1:
            logger.info("=" * 30 + "\nComputing combined_all (sfxl all, pitts30, msls-query, amstertime)...")
            _compute_and_log_combined(combined_all, "combined_all", "ece_combined_all", "Eval_combined_all")

        # --- Other calculations in comment as requested ---
        """
        logger.info("=" * 30 + "\nComputing combined_all ECE across all datasets...")
        _compute_and_log_combined(
            all_agg_data,
            output_subdir="combined_all",
            plot_prefix="ece_combined_all",
            wandb_prefix="Eval_combined_all",
        )
        # ... other combined views ...
        """
    elif len(all_agg_data) <= 1:
        logger.info("Only 1 dataset evaluated; skipping combined ECE.")

    logger.info("=" * 30 + "\nAll processes finished.")
    wandb_utils.finish_run(cfg)