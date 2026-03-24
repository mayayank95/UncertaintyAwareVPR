import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from configs.runtime import build_config_and_datasets, init_model, init_wandb, log_wandb_images
from data.train_dataset import TrainDataset
from data.test_dataset import TestDataset
from eval_metrics import visualizations
from eval_metrics.eval_ece_sh import compute_ece, _cal_mapk
from eval_metrics.uncertainty import compute_uncertainty_correlation, compute_uncertainty_statistics
from losses import uncertainty_utils
from utils import commons, wandb_utils

logger = logging.getLogger(__name__)


def _compute_val_gnll(args, all_descriptors, all_variances, positives_per_query, num_database):
    """Validation uncertainty (GNLL) loss: same formula as train (query vs first positive). Returns float or None."""
    if positives_per_query is None:
        return None
    valid = [(i, pos[0]) for i, pos in enumerate(positives_per_query) if len(pos) > 0]
    if len(valid) == 0:
        return None
    q_idx = np.array([num_database + i for i, _ in valid])
    db_idx = np.array([idx for _, idx in valid])
    q = torch.from_numpy(all_descriptors[q_idx])
    db = torch.from_numpy(all_descriptors[db_idx])
    qv = torch.from_numpy(all_variances[q_idx])
    q_norm = F.normalize(q, p=2, dim=1)
    db_norm = F.normalize(db, p=2, dim=1)
    loss = uncertainty_utils.compute_uncertainty_loss(
        q_norm, db_norm, qv,
        loss_type=args.get("uncertainty_loss", "gaussian_nll"),
        lambda_=args.get("uncertainty_lambda", 1.0),
        gnll_mu_scale_mode=args.get("gnll_mu_scale_mode", "sqrt_dim"),
        gnll_mu_scale_value=args.get("gnll_mu_scale_value", 1.0),
    )
    return loss.item()


def _loss_tokens(args) -> list:
    losses = args.get("losses") or []
    if isinstance(losses, list):
        return [str(x).strip().lower() for x in losses if str(x).strip()]
    return [s.strip().lower() for s in str(losses).split(",") if s.strip()]


def _compute_val_ce(args, all_descriptors, num_database, test_ds, train_group, classifier, device):
    """CosFace CE on val queries: MarginCosine logits + CrossEntropy, same as training.

    Labels are CosPlace class indices in the current training group (from query path UTM + heading).
    Queries whose class is not in this group are skipped."""
    if train_group is None or classifier is None:
        return None
    if "ce" not in _loss_tokens(args):
        return None
    class_to_idx = {cid: i for i, cid in enumerate(train_group.classes_ids)}
    if not class_to_idx:
        return None
    M, alpha, N, L = args["M"], args["alpha"], args["N"], args["L"]
    rows = []
    targets = []
    for j in range(test_ds.num_queries):
        path = test_ds.queries_paths[j]
        parts = path.split("@")
        if len(parts) <= 9:
            continue
        try:
            utm_east = float(parts[1])
            utm_north = float(parts[2])
            heading = float(parts[9])
        except (ValueError, IndexError):
            continue
        class_id, _ = TrainDataset.get__class_id__group_id(utm_east, utm_north, heading, M, alpha, N, L)
        if class_id not in class_to_idx:
            continue
        local_idx = class_to_idx[class_id]
        q = torch.from_numpy(all_descriptors[num_database + j].copy()).float().unsqueeze(0)
        rows.append(q)
        targets.append(local_idx)
    if not rows:
        return None
    q_batch = torch.cat(rows, dim=0).to(device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)
    ce_criterion = nn.CrossEntropyLoss()
    clf = classifier.to(device)
    output = clf(q_batch, targets_t)
    loss = ce_criterion(output, targets_t)
    clf.cpu()
    return float(loss.item())


def _combine_val_loss(args, val_ce: Optional[float], val_gnll: Optional[float]) -> Optional[float]:
    """Total validation loss matching training: CE + uncertainty (when both in losses), else the active part."""
    lt = _loss_tokens(args)
    has_ce = "ce" in lt
    has_u = "uncertainty" in lt and args.get("model_mode") == "uncertainty"
    if not has_ce and not has_u:
        return None
    if has_ce and has_u:
        if val_ce is None or val_gnll is None:
            return None
        return float(val_ce) + float(val_gnll)
    if has_ce:
        return float(val_ce) if val_ce is not None else None
    return float(val_gnll) if val_gnll is not None else None


def eval_dataset(args, model, device, dataset_name, eval_ds_path, wandb_step=None, log_dataset_info=True,
                 classifiers=None, current_group_num=None, groups=None):
    """
    Evaluates the model on a single dataset.
    Extracts features once, then computes Recalls and Uncertainty metrics.
    log_dataset_info: if True, log dataset size (e.g. "Testing on ..."); set False when called every epoch from train.
    classifiers, groups: optional; from train.py they enable val CE term in combined val_loss.
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

    # Validation uncertainty loss (for early stopping when backbone frozen)
    val_gnll = None
    if args.get("model_mode") == "uncertainty" and args.get("use_labels") and positives_per_query is not None:
        val_gnll = _compute_val_gnll(args, all_descriptors, all_variances, positives_per_query, test_ds.num_database)

    # Validation CE (CosFace + cross-entropy): same as train for queries in the current group's classes
    val_ce = None
    gn = int(args.get("groups_num") or 1)
    cg = (wandb_step if wandb_step is not None else 0) % gn
    if (
        args.get("use_labels")
        and classifiers is not None
        and groups is not None
        and cg < len(groups)
        and "ce" in _loss_tokens(args)
    ):
        val_ce = _compute_val_ce(
            args, all_descriptors, test_ds.num_database, test_ds, groups[cg], classifiers[cg], device,
        )

    val_loss = _combine_val_loss(args, val_ce, val_gnll)

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
        if val_loss is not None:
            msg += f", val_loss = {val_loss:.4f}"
            if val_ce is not None and val_gnll is not None:
                msg += f" (ce={val_ce:.4f}+unc={val_gnll:.4f})"
            elif val_ce is not None:
                msg += f" (ce={val_ce:.4f})"
            elif val_gnll is not None:
                msg += f" (unc={val_gnll:.4f})"
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
    if val_loss is not None:
        wandb_metrics[f"{prefix}val_loss"] = float(val_loss)
        if wandb_step is not None:
            wandb_metrics["val/loss"] = float(val_loss)
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

    return recalls, recalls_str, map_at_k, uncertainty_corr, mean_query_variance, std_query_variance, min_query_variance, max_query_variance, wandb_metrics, wandb_images, val_loss

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

        recalls, r_str, map_at_k, corr, mean_var, std_var, min_var, max_var, eval_wb, wandb_images, _ = eval_dataset(
            cfg, model, device, name, datasets_paths[name]["test"], wandb_step=None
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