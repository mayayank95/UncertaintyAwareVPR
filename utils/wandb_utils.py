"""W&B helpers for training and evaluation: epoch/dataset logging and run teardown."""
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from configs.runtime import log_wandb, save_wandb_logs


def _recall_values(cfg: Dict[str, Any]) -> List[int]:
    return cfg.get("recall_values", [1, 5, 10, 20])


def _add_recall_metrics(
    metrics: Dict[str, Any], prefix: str, recalls: np.ndarray, recall_values: List[int]
) -> None:
    """Add recall@k metrics to metrics dict. prefix e.g. 'val/' or 'eval/sf_xl/'."""
    for i, k in enumerate(recall_values):
        if i < len(recalls):
            metrics[f"{prefix}recall@{k}"] = float(recalls[i])


def _add_uncertainty_metrics(
    metrics: Dict[str, Any],
    prefix: str,
    uncertainty_corr: Optional[float],
    mean_variance: Optional[float],
    min_variance: Optional[float],
    max_variance: Optional[float],
) -> None:
    """Add uncertainty/variance metrics to metrics dict."""
    if uncertainty_corr is not None:
        metrics[f"{prefix}uncertainty_correlation"] = float(uncertainty_corr)
    if mean_variance is not None:
        metrics[f"{prefix}mean_variance"] = float(mean_variance)
    if min_variance is not None:
        metrics[f"{prefix}min_variance"] = float(min_variance)
    if max_variance is not None:
        metrics[f"{prefix}max_variance"] = float(max_variance)


def _merge_images_into_metrics(
    metrics: Dict[str, Any], images: Optional[Dict[str, Path]]
) -> None:
    """Add wandb.Image entries for existing paths. Modifies metrics in place."""
    if not images:
        return
    import wandb as _wandb
    for key, img_path in images.items():
        if Path(img_path).exists():
            metrics[key] = _wandb.Image(str(img_path))


def log_train_epoch(
    cfg: Dict[str, Any],
    epoch_num: int,
    recalls: np.ndarray,
    best_val_recall1: float,
    active_losses: List[str],
    epoch_variances: Optional[List[float]],
    epoch_losses: List[float],
    epoch_losses_ce: Optional[List[float]],
    epoch_losses_gnll: Optional[List[float]],
    uncertainty_corr: Optional[float],
    mean_query_variance: Optional[float],
    min_query_variance: Optional[float],
    max_query_variance: Optional[float],
    eval_wandb_metrics: Dict[str, Any],
    eval_wandb_images: Optional[Dict[str, Path]],
) -> None:
    """Build epoch metrics (scalars + images) and log to W&B. No-op if use_wandb is False."""
    if not cfg.get("use_wandb"):
        return
    rv = _recall_values(cfg)
    metrics: Dict[str, Any] = {"epoch": epoch_num, "val/best_recall@1": float(best_val_recall1)}
    _add_recall_metrics(metrics, "val/", recalls, rv)
    if "uncertainty" in active_losses and epoch_variances:
        metrics["train/mean_variance"] = float(np.mean(epoch_variances))
        metrics["train/std_variance"] = float(np.std(epoch_variances))
        metrics["train/min_variance"] = float(np.min(epoch_variances))
        metrics["train/max_variance"] = float(np.max(epoch_variances))
    if active_losses:
        metrics["train/loss"] = float(np.mean(epoch_losses))
    if "ce" in active_losses and epoch_losses_ce:
        metrics["train/loss_ce"] = float(np.mean(epoch_losses_ce))
    if "uncertainty" in active_losses and epoch_losses_gnll:
        metrics["train/loss_uncertainty"] = float(np.mean(epoch_losses_gnll))
    _add_uncertainty_metrics(
        metrics, "val/",
        uncertainty_corr, mean_query_variance, min_query_variance, max_query_variance,
    )
    metrics.update(eval_wandb_metrics)
    _merge_images_into_metrics(metrics, eval_wandb_images)
    log_wandb(metrics, step=epoch_num)


def log_eval_dataset(
    cfg: Dict[str, Any],
    dataset_name: str,
    recalls: np.ndarray,
    uncertainty_corr: Optional[float],
    mean_variance: Optional[float],
    min_variance: Optional[float],
    max_variance: Optional[float],
    eval_wandb_metrics: Dict[str, Any],
) -> None:
    """Log evaluation metrics for one dataset to W&B. No-op if use_wandb is False."""
    if not cfg.get("use_wandb"):
        return
    prefix = f"eval/{dataset_name}/"
    rv = _recall_values(cfg)
    metrics: Dict[str, Any] = {}
    _add_recall_metrics(metrics, prefix, recalls, rv)
    _add_uncertainty_metrics(
        metrics, prefix,
        uncertainty_corr, mean_variance, min_variance, max_variance,
    )
    metrics.update(eval_wandb_metrics)
    log_wandb(metrics)


def finish_run(cfg: Dict[str, Any]) -> None:
    """Upload log files and finish W&B run. No-op if use_wandb is False. Use for both train and eval."""
    if not cfg.get("use_wandb"):
        return
    save_wandb_logs(cfg.get("log_dir"))
    import wandb
    wandb.finish()


def finish_train_run(cfg: Dict[str, Any]) -> None:
    """Alias for finish_run (train entrypoint)."""
    finish_run(cfg)
