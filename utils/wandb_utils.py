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


def _merge_images_into_metrics(
    metrics: Dict[str, Any], images: Optional[Dict[str, Path]]
) -> None:
    """Add wandb.Image entries for existing paths. Modifies metrics in place."""
    if not images:
        return
    import wandb as _wandb
    for key, img_path in images.items():
        if isinstance(img_path, list):
            valid_images = [Path(p) for p in img_path if Path(p).exists()]
            if valid_images:
                metrics[key] = [_wandb.Image(str(p)) for p in valid_images]
        else:
            if Path(img_path).exists():
                metrics[key] = _wandb.Image(str(img_path))


def _train_epoch_images_for_sections(images: Optional[Dict[str, Path]]) -> Dict[str, Path]:
    """Map eval images to train panel sections: val/variance_distribution, ece/ece_plot (no eval/ section)."""
    if not images:
        return {}
    out = {}
    for key, path in images.items():
        if isinstance(path, list):
            if "predictions" in key:
                out["val/predictions"] = path
            continue
            
        p = Path(path).resolve()
        if not p.exists():
            continue
        if "variance_distribution" in key:
            out["val/variance_distribution"] = p
        elif "ece_plot" in key:
            out["ece/ece_plot"] = p
        elif "uncertainty_correlation_scatter" in key:
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
    """Build epoch metrics (scalars + images) and log to W&B. No-op if use_wandb is False.

    Panel ordering (dict insertion order determines initial W&B layout):
      train/  — losses (ce, uncertainty, total), variance stats (min, max, mean, std)
      val/    — losses, recalls, variance stats, variance distribution plot
      ece/    — recalls, ece plot
    """
    if not cfg.get("use_wandb"):
        return
    rv = _recall_values(cfg)
    metrics: Dict[str, Any] = {}
    metrics["epoch"] = epoch_num

    # ── train/ ── losses (individual then total), variance statistics
    if "ce" in active_losses and epoch_losses_ce:
        metrics["train/loss_ce"] = float(np.mean(epoch_losses_ce))
    if "uncertainty" in active_losses and epoch_losses_gnll:
        metrics["train/loss_uncertainty"] = float(np.mean(epoch_losses_gnll))
    if active_losses:
        metrics["train/loss"] = float(np.mean(epoch_losses))
    if "uncertainty" in active_losses and epoch_variances:
        metrics["train/variance_min"] = float(np.min(epoch_variances))
        metrics["train/variance_max"] = float(np.max(epoch_variances))
        metrics["train/variance_mean"] = float(np.mean(epoch_variances))
        metrics["train/variance_std"] = float(np.std(epoch_variances))

    # ── val/ ── losses, recalls, variance statistics, plots
    if "val/loss" in eval_wandb_metrics:
        metrics["val/loss"] = float(eval_wandb_metrics["val/loss"])
    metrics["val/best_recall_01"] = float(best_val_recall1)
    _add_recall_metrics(metrics, "val/", recalls, rv)
    _add_map_metrics(metrics, "val/", map_at_k, rv)
    if uncertainty_corr is not None:
        metrics["val/uncertainty_correlation"] = float(uncertainty_corr)
    if min_query_variance is not None:
        metrics["val/variance_min"] = float(min_query_variance)
    if max_query_variance is not None:
        metrics["val/variance_max"] = float(max_query_variance)
    if mean_query_variance is not None:
        metrics["val/variance_mean"] = float(mean_query_variance)
    if std_query_variance is not None:
        metrics["val/variance_std"] = float(std_query_variance)
    epoch_images = _train_epoch_images_for_sections(eval_wandb_images)
    if "val/variance_distribution" in epoch_images and epoch_images["val/variance_distribution"] is not None:
        _merge_images_into_metrics(metrics, {"val/variance_distribution": epoch_images["val/variance_distribution"]})
    if "val/uncertainty_correlation_scatter" in epoch_images and epoch_images["val/uncertainty_correlation_scatter"] is not None:
        _merge_images_into_metrics(metrics, {"val/uncertainty_correlation_scatter": epoch_images["val/uncertainty_correlation_scatter"]})
    if "val/predictions" in epoch_images and epoch_images["val/predictions"]:
        _merge_images_into_metrics(metrics, {"val/predictions": epoch_images["val/predictions"]})
    _val_handled = {"val/loss_total", "val/loss"}
    for k, v in eval_wandb_metrics.items():
        if not k.startswith("eval/") and k.startswith("val/") and k not in metrics and k not in _val_handled:
            metrics[k] = v

    # ── ece/ ── recalls, then ece plot
    for k in sorted(rv):
        key = f"ece/recall_{k:02d}"
        metrics[key] = float(eval_wandb_metrics.get(key, 0.0))
    for k in sorted(rv):
        key = f"ece/map_{k:02d}"
        metrics[key] = float(eval_wandb_metrics.get(key, 0.0))
    metrics["ece/ap"] = float(eval_wandb_metrics.get("ece/ap", 0.0))
    if "ece/ece_plot" in epoch_images and epoch_images["ece/ece_plot"] is not None:
        _merge_images_into_metrics(metrics, {"ece/ece_plot": epoch_images["ece/ece_plot"]})

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
    """Log evaluation metrics and images for one dataset to W&B.

    Panel ordering per dataset (dict insertion order determines initial W&B layout):
      Eval_{name}/  — recalls, ECE values, ECE plot, variance distribution,
                      variance statistics, predictions
    """
    if not cfg.get("use_wandb"):
        return
    import wandb as _wandb

    prefix = f"Eval_{dataset_name}"
    rv = _recall_values(cfg)
    metrics: Dict[str, Any] = {}

    def _html_table(headers, rows):
        style = (
            "style='border-collapse:collapse;width:100%;font-family:monospace;font-size:16px;'"
        )
        th_style = "style='border:1px solid #555;padding:8px;background:#2a2a2a;color:#eee;text-align:center;'"
        td_style = "style='border:1px solid #555;padding:8px;text-align:center;'"
        td_label = "style='border:1px solid #555;padding:8px;text-align:left;font-weight:bold;'"
        html = f"<table {style}><tr>"
        for h in headers:
            html += f"<th {th_style}>{h}</th>"
        html += "</tr>"
        for row in rows:
            html += "<tr>"
            for i, cell in enumerate(row):
                s = td_label if i == 0 else td_style
                if isinstance(cell, float):
                    cell = f"{cell:.2f}"
                html += f"<td {s}>{cell}</td>"
            html += "</tr>"
        html += "</table>"
        return _wandb.Html(html)

    # ── 1. Recalls ──
    ret_headers = ["Metric"] + [f"@{k}" for k in sorted(rv)]
    ret_rows = []
    recall_row = ["Recall"]
    for k in sorted(rv):
        i = rv.index(k)
        val = float(recalls[i]) if i < len(recalls) else 0.0
        recall_row.append(val)
        metrics[f"{prefix}/{_recall_key(k)}"] = val
    ret_rows.append(recall_row)
    if map_at_k is not None:
        map_row = ["mAP"]
        for k in sorted(rv):
            i = rv.index(k)
            val = float(map_at_k[i]) if i < len(map_at_k) else 0.0
            map_row.append(val)
            metrics[f"{prefix}/{_map_key(k)}"] = val
        ret_rows.append(map_row)
    metrics[f"{prefix}/retrieval_metrics"] = _html_table(ret_headers, ret_rows)

    # ── 2. ECE values ──
    ece_rows = []
    has_ece_recall = any(f"Eval_{dataset_name}/ece_recall_" in k for k in eval_wandb_metrics)
    has_ece_map = any(f"Eval_{dataset_name}/ece_map_" in k for k in eval_wandb_metrics)
    if has_ece_recall:
        row = ["ECE Recall"]
        for k in sorted(rv):
            key = f"Eval_{dataset_name}/ece_recall_{k:02d}"
            val = float(eval_wandb_metrics.get(key, 0.0))
            row.append(val)
            metrics[key] = val
        ece_rows.append(row)
    if has_ece_map:
        row = ["ECE mAP"]
        for k in sorted(rv):
            key = f"Eval_{dataset_name}/ece_map_{k:02d}"
            val = float(eval_wandb_metrics.get(key, 0.0))
            row.append(val)
            metrics[key] = val
        ece_rows.append(row)
    ece_ap_key = f"Eval_{dataset_name}/ece_ap"
    if ece_ap_key in eval_wandb_metrics:
        val = float(eval_wandb_metrics[ece_ap_key])
        ece_rows.append(["ECE AP", val])
        metrics[ece_ap_key] = val
    if ece_rows:
        ece_headers = ["Metric"] + [f"@{k}" for k in sorted(rv)]
        metrics[f"{prefix}/ece_metrics"] = _html_table(ece_headers, ece_rows)

    # ── 3. ECE plot ──
    if eval_wandb_images:
        ece_key = f"Eval_{dataset_name}/ece_plot"
        if ece_key in eval_wandb_images:
            p = Path(eval_wandb_images[ece_key])
            if p.exists():
                metrics[f"{prefix}/ece_plot"] = _wandb.Image(str(p))

    # ── 4. Variance distribution plot ──
    if eval_wandb_images:
        var_dist_key = f"Eval_{dataset_name}/variance_distribution"
        if var_dist_key in eval_wandb_images:
            p = Path(eval_wandb_images[var_dist_key])
            if p.exists():
                metrics[f"{prefix}/variance_distribution"] = _wandb.Image(str(p))

    # ── 5. Variance statistics ──
    var_headers = []
    var_row = []
    if uncertainty_corr is not None:
        var_headers.append("Correlation")
        var_row.append(float(uncertainty_corr))
        metrics[f"{prefix}/uncertainty_correlation"] = float(uncertainty_corr)
    if min_variance is not None:
        var_headers.append("Min")
        var_row.append(float(min_variance))
        metrics[f"{prefix}/variance_min"] = float(min_variance)
    if max_variance is not None:
        var_headers.append("Max")
        var_row.append(float(max_variance))
        metrics[f"{prefix}/variance_max"] = float(max_variance)
    if mean_variance is not None:
        var_headers.append("Mean")
        var_row.append(float(mean_variance))
        metrics[f"{prefix}/variance_mean"] = float(mean_variance)
    if std_variance is not None:
        var_headers.append("Std")
        var_row.append(float(std_variance))
        metrics[f"{prefix}/variance_std"] = float(std_variance)
    if var_headers:
        metrics[f"{prefix}/variance_statistics"] = _html_table(var_headers, [var_row])
    if eval_wandb_images:
        scatter_key = f"Eval_{dataset_name}/uncertainty_correlation_scatter"
        if scatter_key in eval_wandb_images:
            p = Path(eval_wandb_images[scatter_key])
            if p.exists():
                metrics[f"{prefix}/uncertainty_correlation_scatter"] = _wandb.Image(str(p))

    # ── 6. Predictions visualization ──
    if eval_wandb_images:
        preds_key = f"Eval_{dataset_name}/predictions"
        if preds_key in eval_wandb_images:
            img_list = eval_wandb_images[preds_key]
            if isinstance(img_list, list):
                valid = [Path(p) for p in img_list if Path(p).exists()]
                if valid:
                    metrics[f"{prefix}/predictions"] = [_wandb.Image(str(p)) for p in valid]

    # Forward any remaining scalar metrics not yet handled
    for k, v in eval_wandb_metrics.items():
        if k not in metrics:
            metrics[k] = v

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
