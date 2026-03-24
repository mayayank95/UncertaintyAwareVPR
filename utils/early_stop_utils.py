"""Early-stop metric parsing and reading values from eval_dataset W&B metric dicts."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

_ECE_RECALL_ALIAS = {"ece_recall": "ece_recall_01"}

_ALLOWED = frozenset({"recall", "ece_recall_01", "ece_recall_05", "ece_recall_10", "val_loss"})


def canonical_early_stop_metrics(raw: Any) -> List[str]:
    """Parse CLI/config value into ordered unique canonical metric ids."""
    if raw is None:
        return ["recall"]
    if isinstance(raw, list):
        tokens = [str(x).strip().lower().replace("-", "_") for x in raw if str(x).strip()]
    else:
        tokens = [s.strip().lower().replace("-", "_") for s in str(raw).split(",") if s.strip()]
    if not tokens:
        return ["recall"]
    out: List[str] = []
    for t in tokens:
        t = _ECE_RECALL_ALIAS.get(t, t)
        if t not in _ALLOWED:
            raise ValueError(
                f"Invalid early_stop_metric token {t!r}. "
                f"Allowed: {sorted(_ALLOWED)} (use ece_recall for R@1 ECE)."
            )
        if t not in out:
            out.append(t)
    return out


def recall_values_needed_for_metrics(metrics: List[str]) -> List[int]:
    """K values that must appear in recall_values for ECE @K early stops."""
    need: List[int] = []
    for m in metrics:
        if m.startswith("ece_recall_"):
            need.append(int(m.rsplit("_", 1)[-1]))
    return need


def best_model_filename(metric: str, metrics: List[str]) -> str:
    """Filename for a metric-specific best checkpoint (weights only)."""
    if len(metrics) == 1 and metrics[0] == "recall":
        return "best_model.pth"
    return f"best_model_{metric}.pth"


def get_metric_value(
    metric: str,
    *,
    recalls: Any,
    recall_values: List[int],
    eval_wandb_metrics: Optional[Dict[str, Any]],
    dataset_name: str,
    val_gnll: Optional[float],
) -> Optional[float]:
    """Scalar for this epoch, or None if unavailable."""
    if metric == "recall":
        if recalls is None or len(recalls) == 0:
            return None
        return float(recalls[0])
    if metric.startswith("ece_recall_"):
        k = int(metric.rsplit("_", 1)[-1])
        if k not in recall_values:
            return None
        idx = recall_values.index(k)
        flat = f"ece/recall_{k:02d}"
        if eval_wandb_metrics and flat in eval_wandb_metrics:
            return float(eval_wandb_metrics[flat])
        pref = f"Eval_{dataset_name}/ece_recall_{k:02d}"
        if eval_wandb_metrics and pref in eval_wandb_metrics:
            return float(eval_wandb_metrics[pref])
        return None
    if metric == "val_loss":
        if val_gnll is not None:
            return float(val_gnll)
        if eval_wandb_metrics:
            if "val/loss_uncertainty" in eval_wandb_metrics:
                return float(eval_wandb_metrics["val/loss_uncertainty"])
            pref = f"Eval_{dataset_name}/val_loss_uncertainty"
            if pref in eval_wandb_metrics:
                return float(eval_wandb_metrics[pref])
        return None
    return None


def initial_best_for_metric(metric: str) -> float:
    if metric == "recall":
        return 0.0
    return float("inf")


def is_improvement(metric: str, current: float, best_so_far: float) -> bool:
    if metric == "recall":
        return current > best_so_far
    return current < best_so_far
