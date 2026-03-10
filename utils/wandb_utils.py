"""W&B helpers for training and evaluation: epoch/dataset logging and run teardown."""
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from configs.runtime import log_wandb, save_wandb_logs


def _recall_values(cfg: Dict[str, Any]) -> List[int]:
    return cfg.get("recall_values", [1, 5, 10, 20])


def _recall_key(k: int) -> str:
    """Format recall_01, recall_05, ... so all recalls group together in W&B."""
    return f"recall_{k:02d}"


def _map_key(k: int) -> str:
    """Format map_01, map_05, ... so all maps group together in W&B."""
    return f"map_{k:02d}"


def _add_recall_metrics(
    metrics: Dict[str, Any], prefix: str, recalls: np.ndarray, recall_values: List[int]
) -> None:
    """Add recall@k metrics to metrics dict (order: R@1, R@5, R@10, ...). prefix e.g. 'val/' or 'eval/sf_xl/'."""
    for k in sorted(recall_values):
        i = recall_values.index(k)
        if i < len(recalls):
            metrics[f"{prefix}{_recall_key(k)}"] = float(recalls[i])


def _add_map_metrics(
    metrics: Dict[str, Any],
    prefix: str,
    map_at_k: Optional[List[float]],
    recall_values: List[int],
) -> None:
    """Add mAP@k metrics to metrics dict (order: mAP@1, mAP@5, ...). prefix e.g. 'val/' or 'eval/sf_xl/'."""
    if map_at_k is None:
        return
    for k in sorted(recall_values):
        i = recall_values.index(k)
        if i < len(map_at_k):
            metrics[f"{prefix}{_map_key(k)}"] = float(map_at_k[i])


def _add_uncertainty_metrics(
    metrics: Dict[str, Any],
    prefix: str,
    uncertainty_corr: Optional[float],
    mean_variance: Optional[float],
    std_variance: Optional[float],
    min_variance: Optional[float],
    max_variance: Optional[float],
    variance_subprefix: bool = False,
) -> None:
    """Add uncertainty/variance metrics to metrics dict.
    If variance_subprefix is True, use variance_mean, variance_std, variance_min, variance_max so they group together in W&B."""
    if uncertainty_corr is not None:
        metrics[f"{prefix}uncertainty_correlation"] = float(uncertainty_corr)
    if variance_subprefix:
        if mean_variance is not None:
            metrics[f"{prefix}variance_mean"] = float(mean_variance)
        if std_variance is not None:
            metrics[f"{prefix}variance_std"] = float(std_variance)
        if min_variance is not None:
            metrics[f"{prefix}variance_min"] = float(min_variance)
        if max_variance is not None:
            metrics[f"{prefix}variance_max"] = float(max_variance)
    else:
        if mean_variance is not None:
            metrics[f"{prefix}mean_variance"] = float(mean_variance)
        if std_variance is not None:
            metrics[f"{prefix}std_variance"] = float(std_variance)
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


def _train_epoch_images_for_sections(images: Optional[Dict[str, Path]]) -> Dict[str, Path]:
    """Map eval images to train panel sections: val/variance_distribution, ece/ece_plot (no eval/ section)."""
    if not images:
        return {}
    out = {}
    for key, path in images.items():
        p = Path(path).resolve()
        if not p.exists():
            continue
        if "variance_distribution" in key:
            out["val/variance_distribution"] = p
        if "ece_plot" in key:
            out["ece/ece_plot"] = p
        if "uncertainty_correlation_scatter" in key:
            out["val/uncertainty_correlation_scatter"] = p
    return out


def log_train_epoch(
    cfg: Dict[str, Any],
    epoch_num: int,
    recalls: np.ndarray,
    map_at_k: Optional[List[float]],
    best_val_recall1: float,
    active_losses: List[str],
    epoch_variances: Optional[List[float]],
    epoch_losses: List[float],
    epoch_losses_ce: Optional[List[float]],
    epoch_losses_gnll: Optional[List[float]],
    uncertainty_corr: Optional[float],
    mean_query_variance: Optional[float],
    std_query_variance: Optional[float],
    min_query_variance: Optional[float],
    max_query_variance: Optional[float],
    eval_wandb_metrics: Dict[str, Any],
    eval_wandb_images: Optional[Dict[str, Path]],
) -> None:
    """Build epoch metrics (scalars + images) and log to W&B. No-op if use_wandb is False."""
    if not cfg.get("use_wandb"):
        return
    rv = _recall_values(cfg)
    metrics: Dict[str, Any] = {}
    metrics["epoch"] = epoch_num
    # train/ first
    if active_losses:
        metrics["train/loss"] = float(np.mean(epoch_losses))
    if "ce" in active_losses and epoch_losses_ce:
        metrics["train/loss_ce"] = float(np.mean(epoch_losses_ce))
    if "uncertainty" in active_losses and epoch_losses_gnll:
        metrics["train/loss_uncertainty"] = float(np.mean(epoch_losses_gnll))
    if "uncertainty" in active_losses and epoch_variances:
        metrics["train/mean_variance"] = float(np.mean(epoch_variances))
        metrics["train/std_variance"] = float(np.std(epoch_variances))
        metrics["train/min_variance"] = float(np.min(epoch_variances))
        metrics["train/max_variance"] = float(np.max(epoch_variances))
    # val/ — order: images (variance_distribution, correlation_scatter), recalls, loss(es), maps, rest
    # 1) Images first
    epoch_images = _train_epoch_images_for_sections(eval_wandb_images)
    if "val/variance_distribution" in epoch_images and epoch_images["val/variance_distribution"] is not None:
        _merge_images_into_metrics(metrics, {"val/variance_distribution": epoch_images["val/variance_distribution"]})
    if "val/uncertainty_correlation_scatter" in epoch_images and epoch_images["val/uncertainty_correlation_scatter"] is not None:
        _merge_images_into_metrics(metrics, {"val/uncertainty_correlation_scatter": epoch_images["val/uncertainty_correlation_scatter"]})
    # 2) Recalls (all together)
    metrics["val/best_recall_01"] = float(best_val_recall1)
    _add_recall_metrics(metrics, "val/", recalls, rv)
    # 3) Loss(es): use numeric prefix so W&B sorts them in order (01_ce, 02_uncertainty, 03_total)
    val_loss_parts: List[float] = []
    if "val/loss_ce" in eval_wandb_metrics:
        v = float(eval_wandb_metrics["val/loss_ce"])
        metrics["val/loss_01_ce"] = v
        val_loss_parts.append(v)
    elif "ce" in active_losses:
        metrics["val/loss_01_ce"] = 0.0
        val_loss_parts.append(0.0)
    if "val/loss_uncertainty" in eval_wandb_metrics:
        v = float(eval_wandb_metrics["val/loss_uncertainty"])
        metrics["val/loss_02_uncertainty"] = v
        val_loss_parts.append(v)
    if val_loss_parts:
        metrics["val/loss_03_total"] = sum(val_loss_parts)
    # 4) Maps (all together)
    _add_map_metrics(metrics, "val/", map_at_k, rv)
    # 5) Rest (uncertainty, then variance metrics grouped: mean, std, min, max)
    _add_uncertainty_metrics(
        metrics, "val/",
        uncertainty_corr, mean_query_variance, std_query_variance, min_query_variance, max_query_variance,
        variance_subprefix=True,
    )
    _val_loss_keys_skip = {"val/loss_ce", "val/loss_uncertainty", "val/loss_total", "val/loss"}
    for k, v in eval_wandb_metrics.items():
        if not k.startswith("eval/") and k.startswith("val/") and k not in metrics and k not in _val_loss_keys_skip:
            metrics[k] = v
    # ece/ — ECE plot first, then scalar metrics
    if "ece/ece_plot" in epoch_images and epoch_images["ece/ece_plot"] is not None:
        _merge_images_into_metrics(metrics, {"ece/ece_plot": epoch_images["ece/ece_plot"]})
    rv = _recall_values(cfg)
    for k in sorted(rv):
        key = f"ece/recall_{k:02d}"
        metrics[key] = float(eval_wandb_metrics.get(key, 0.0))
    for k in sorted(rv):
        key = f"ece/map_{k:02d}"
        metrics[key] = float(eval_wandb_metrics.get(key, 0.0))
    metrics["ece/ap"] = float(eval_wandb_metrics.get("ece/ap", 0.0))
    log_wandb(metrics, step=epoch_num)


def log_eval_dataset(
    cfg: Dict[str, Any],
    dataset_name: str,
    recalls: np.ndarray,
    map_at_k: Optional[List[float]],
    uncertainty_corr: Optional[float],
    mean_variance: Optional[float],
    std_variance: Optional[float],
    min_variance: Optional[float],
    max_variance: Optional[float],
    eval_wandb_metrics: Dict[str, Any],
    eval_wandb_images: Optional[Dict[str, Path]] = None,
) -> None:
    """Log evaluation metrics and images for one dataset to W&B. No-op if use_wandb is False.
    Eval-only: no step; just scalars + ECE/variance plots + prediction images under eval/{dataset_name}/."""
    if not cfg.get("use_wandb"):
        return
    prefix = f"eval/{dataset_name}/"
    rv = _recall_values(cfg)
    metrics: Dict[str, Any] = {}
    _add_recall_metrics(metrics, prefix, recalls, rv)
    _add_map_metrics(metrics, prefix, map_at_k, rv)
    _add_uncertainty_metrics(
        metrics, prefix,
        uncertainty_corr, mean_variance, std_variance, min_variance, max_variance,
    )
    metrics.update(eval_wandb_metrics)
    _merge_images_into_metrics(metrics, eval_wandb_images)
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
