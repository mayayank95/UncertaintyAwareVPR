from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from utils.early_stop_utils import canonical_early_stop_metrics, recall_values_needed_for_metrics

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Config-first CLI: JSON config + CLI overrides.")

    # General
    p.add_argument("--config", type=str, default="configs/datasets.json", help="Path to datasets config JSON")
    p.add_argument("--save_config", action="store_true", help="Save merged configuration to logs folder")
    p.add_argument("--data_folder", type=str, default=None, help="Root folder containing raw datasets")
    p.add_argument("--local_data_folder", type=str, default=None, help="Colab: target folder for prepared datasets")
    p.add_argument("--logs_folder", type=str, default=None, help="Folder to save logs")
    p.add_argument("--colab", action="store_true", help="Run in Google Colab mode")
    p.add_argument("--dry_run", action="store_true", help="Print actions without performing file operations")
    p.add_argument("--debug", action="store_true", help="Enable extra debug-only checks and computations.")
    p.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default="UncertaintyAwareVPR", help="W&B project name")
    p.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (default: auto from define function in runtime.py)")

    # Dataset selection
    p.add_argument("--datasets", type=str, default=None, help='Datasets to process (e.g. "all", "sf_xl", or "sf_xl,pitts30k")')
    p.add_argument("--datasets_type", type=str, default="all", help='Dataset splits to use (e.g. "all", "train", "test", "val")')

    # Resume / checkpoint
    p.add_argument("--resume_train", type=str, default=None, help="Path to training checkpoint, e.g. logs/.../last_checkpoint.pth")
    p.add_argument("--resume_model", type=str, default=None, help="Path to model weights, e.g. logs/.../best_model.pth")
    p.add_argument("--ckpt_state_dict_key", type=str, default="model_state_dict", 
                   help="Key for state dict in checkpoint (default: model_state_dict, common alternatives: state_dict)")
    p.add_argument("--load_classifiers", type=str, default=None, help="Path to checkpoint to load and freeze classifier weights from")

    # Model
    p.add_argument("--backbone", type=str, default=None, help="Backbone architecture (e.g. ResNet18, ResNet50, VGG16)")
    p.add_argument("--descriptors_dimension", type=int, default=None, help="Dimension of the output descriptor vector")

    p.add_argument("--method", type=str, default=None, help="Model method (e.g. cosplace, cosplace_pretrained)")
    p.add_argument("--image_size", type=int, default=None, help="Resize images to this size (square)")
    p.add_argument("--train_all_layers", action="store_true", help="Train all backbone layers (default: freeze early layers)")
    p.add_argument("--resize_test_imgs", action="store_true", help="Resize test images to (image_size x image_size)")
    
    # System
    p.add_argument("--device", type=str, default="auto", help="Device to use: 'cuda', 'cpu', or 'auto'")
    p.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers")


    # Evaluation
    p.add_argument("--recall_values", type=int, nargs="+", default=[1, 5, 10, 20], help="Recall@K values to compute")
    p.add_argument("--infer_batch_size", type=int, default=16, help="Batch size for inference (validation and test)")
    p.add_argument("--save_descriptors", action="store_true", help="Save extracted descriptors to disk")
    p.add_argument("--positive_dist_threshold", type=int, default=None, help="Distance in meters for a positive match")
    p.add_argument("--use_labels", action="store_true", help="Use UTM coordinates from image paths for evaluation")
    p.add_argument("--only_recalls", action="store_true", help="Only compute recalls, skip mAP and other metrics for speed")

    # Visualization
    p.add_argument("--num_preds_to_save", type=int, default=3, help="Number of predictions to save per query")
    p.add_argument("--num_queries_to_save", type=int, default=3, help="Number of queries to save predictions for")
    p.add_argument("--save_only_wrong_preds", action="store_true", help="Only save wrongly predicted queries")
    p.add_argument("--save_plots", action="store_true", help="Save evaluation plots (variance distribution, ECE, predictions) for val/test splits")

    # Training
    p.add_argument("--cudnn_benchmark", action="store_true", help="Enable cuDNN benchmark (faster but non-deterministic)")
    p.add_argument("--lr", type=float, default=0.00001, help="Learning rate for model optimizer")
    p.add_argument("--seed", type=int, default=0, help="Random seed (-1 to disable deterministic mode)")
    p.add_argument("--classifiers_lr", type=float, default=0.01, help="Learning rate for classifier optimizers")
    p.add_argument("--batch_size", type=int, default=None, help="Batch size for training DataLoader")
    p.add_argument("--iterations_per_epoch", type=int, default=10000, help="Number of training iterations per epoch")
    p.add_argument(
        "--log_every_n_iterations",
        type=int,
        default=1000,
        help="Log training stats every N iterations within each epoch (rolling mean over last N); 0 disables.",
    )
    p.add_argument("--epochs_num", type=int, default=50, help="Total number of training epochs")
    p.add_argument("--patience", type=int, default=None, help="Patience for early stopping (epochs without improvement)")
    p.add_argument("--disable_early_stop", action="store_true", help="Disable early stopping and always run all epochs")
    p.add_argument("--losses", type=str, default=None, help="Losses to use, comma-separated (e.g. 'ce', 'ce,uncertainty')")

    # Data augmentation
    p.add_argument("--augmentation_device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for data augmentation")
    p.add_argument("--brightness", type=float, default=0.7, help="ColorJitter brightness factor")
    p.add_argument("--contrast", type=float, default=0.7, help="ColorJitter contrast factor")
    p.add_argument("--hue", type=float, default=0.5, help="ColorJitter hue factor")
    p.add_argument("--saturation", type=float, default=0.7, help="ColorJitter saturation factor")
    p.add_argument("--random_resized_crop", type=float, default=0.5, help="RandomResizedCrop minimum scale (max is 1)")

    # CosPlace parameters
    p.add_argument("--M", type=int, default=10, help="Size of the cell in meters")
    p.add_argument("--alpha", type=int, default=30, help="Size of the margin in degrees")
    p.add_argument("--N", type=int, default=5, help="Min number of images per place")
    p.add_argument("--L", type=int, default=2, help="Smoothing for group boundaries")
    p.add_argument("--groups_num", type=int, default=0, help="If set to 0 use N*N groups")
    p.add_argument("--min_images_per_class", type=int, default=10, help="Minimum images per class for a group to be valid")

    # Uncertainty
    p.add_argument("--model_mode", type=str, default=None, help="Model mode: basic or uncertainty")
    p.add_argument("--sigma_dim", type=int, default=None, help="Dimension of the variance output vector")
    p.add_argument("--uncertainty_lambda", type=float, default=1.0, help="Weight for the uncertainty loss")
    p.add_argument("--uncertainty_loss", type=str, default=None, help="Uncertainty loss type: gaussian_nll, gaussian_cosine, or vmf")
    p.add_argument(
        "--gnll_mu_scale_mode",
        type=str,
        default="sqrt_dim",
        choices=["sqrt_dim", "none", "custom"],
        help="Scaling mode for Gaussian NLL mean vectors: sqrt_dim (default), none (1.0), or custom (use --gnll_mu_scale_value).",
    )
    p.add_argument(
        "--gnll_mu_scale_value",
        type=float,
        default=1.0,
        help="Custom scale value for Gaussian NLL mean vectors when --gnll_mu_scale_mode=custom.",
    )
    p.add_argument("--var_head_type", type=str, default="linear",
                   choices=["activation", "linear", "mlp", "separate_agg", "separate_linear_agg", "vmf", "vmf_agg", "stun_head"],
                   help="Variance head type: 'activation' (bare activation, no trainable params), "
                        "'linear' (Linear + activation), 'mlp' (GeM + 2-layer MLP + activation), "
                        "'separate_agg' (deep copy of aggregation + activation), "
                        "'vmf' (Linear(fc_output_dim -> 1) + activation), "
                        "'vmf_agg' (deep copy aggregation + Linear(fc_output_dim -> 1) + activation)")
    p.add_argument("--var_init", action="store_true",
                   help="Initialize variance head for stable start: small weights + bias tuned for ~0.1 initial variance")
    p.add_argument("--variance_activation", type=str, default=None, choices=["softplus", "sigmoid"],
                   help="Activation for variance head output: softplus (positive, unbounded) or sigmoid (bounded [0,1])")
    p.add_argument("--freeze_model", action="store_true", help="Freeze backbone/aggregation, train only the uncertainty head")
    p.add_argument("--head_lr", type=float, default=None,
                   help="LR for head (var_head + final_l2) when freeze_model. Default: 1e-3 when freeze_model, else same as --lr")
    p.add_argument(
        "--early_stop_metric",
        type=str,
        default="recall",
        help="Early-stop metrics separated by commas (,). Example: recall,ece_recall. "
        "Tokens: recall; ece_recall (R@1 ECE); ece_recall_05; ece_recall_10. "
        "Each metric has its own patience counter; each saves best_model_<metric>.pth (best_model.pth when only recall).",
    )
    p.add_argument(
        "--phased_early_stop",
        action="store_true",
        help="Two-phase early stopping: Phase 1 tracks recall only; when recall plateaus "
        "(exhausts patience), Phase 2 activates the ECE metrics with fresh patience. "
        "Requires early_stop_metric to include both recall and at least one ece_recall_* metric.",
    )
    p.add_argument("--debug_var_head_grad", action="store_true",
                   help="Log var_head gradient norms after backward (one line per epoch) to verify gradients flow")
    p.add_argument(
        "--freeze_batchnorm",
        action="store_true",
        help="Freeze BatchNorm layers (set to eval mode) during training to keep running stats fixed.",
    )
    p.add_argument(
        "--ece_metrics",
        type=str,
        default="recall",
        help="ECE metrics to compute (comma-separated). Default: recall only. "
             "Options: recall, map, ap. calibration uses --pairwise_metrics instead.",
    )
    p.add_argument(
        "--skip_detailed_metrics",
        action="store_true",
        help="Skip expensive uncertainty AUC-PR and baseline calculations (L2, PA-score, SUE) in eval.py.",
    )
    p.add_argument(
        "--ece_zoom_threshold",
        type=float,
        default=0.001,
        help="Fraction of data required in the last bin for ECE adaptive zoom (default 0.001 = 0.1%).",
    )
    p.add_argument(
        "--ece_bin_mode",
        type=str,
        default="zoom",
        choices=["zoom", "percentile"],
        help="ECE binning mode: 'zoom' (adaptive zoom, legacy) or 'percentile' (percentile-clamped uniform bins; method-aware clipping). Default: zoom.",
    )
    p.add_argument(
        "--ece_vmf_kappa_floor",
        action="store_true",
        help="Enable vMF kappa flooring (kappa<1 -> 1) before ECE inversion.",
    )
    return p.parse_args()
    

def load_config(path: Path) -> Dict[str, Any]:
    path = path.expanduser()
    if not path.exists():
        logger.error(f"Config not found: {path}") # Log error before raising
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def merge_cfg_with_cli(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge rule: merged = cfg + (CLI overrides for keys where CLI value is not None)."""
    cli = vars(args).copy()
    cli.pop("config", None)

    overrides = {k: v for k, v in cli.items() if v is not None}
    merged = dict(cfg)
    merged.update(overrides)
    return merged


def normalize(merged: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common fields to expected types/format."""
    out = dict(merged)

    # Paths: store as strings in config, but normalize to expanded string paths
    for k in ("data_folder", "local_data_folder", "logs_folder", "resume_train", "resume_model"):
        if k in out and out[k] is not None:
            out[k] = str(Path(out[k]).expanduser())

    for element in ("datasets", "datasets_type", "losses"):
        if element in out and out[element] is not None:
            v = out[element]
            if isinstance(v, list):
                continue
            v = str(v).strip()
            if v.lower() == "all":
                out[element] = "all"
            else:
                out[element] = [s.strip() for s in v.split(",") if s.strip()]

    if out.get("groups_num") == 0 and "N" in out:
        out["groups_num"] = out["N"] * out["N"]



    if "ece_metrics" in out and out["ece_metrics"] is not None:
        v = out["ece_metrics"]
        if isinstance(v, list):
            out["ece_metrics"] = [str(x).strip().lower() for x in v if str(x).strip()]
        else:
            out["ece_metrics"] = [s.strip().lower() for s in str(v).split(",") if s.strip()]

    # if "pairwise_metrics" in out and out["pairwise_metrics"] is not None:
    #     v = out["pairwise_metrics"]
    #     if isinstance(v, list):
    #         pm = [str(x).strip().lower() for x in v if str(x).strip()]
    #     else:
    #         pm = [s.strip().lower() for s in str(v).split(",") if s.strip()]
    #     allowed = frozenset({"l2", "jrl"})
    #     for x in pm:
    #         if x not in allowed:
    #             raise ValueError(f"pairwise_metrics: unknown entry {x!r}; allowed: l2, jrl.")
    #     seen = set()
    #     canon = []
    #     for x in pm:
    #         if x in seen:
    #             continue
    #         seen.add(x)
    #         canon.append(x)
    #     out["pairwise_metrics"] = canon

    raw_es = out.get("early_stop_metric", "recall")
    out["early_stop_metrics"] = canonical_early_stop_metrics(raw_es)
    for m in out["early_stop_metrics"]:
        if m.startswith("ece_recall_"):
            if out.get("model_mode") != "uncertainty":
                raise ValueError(f"early_stop_metric {m} requires model_mode=uncertainty.")
            if not out.get("use_labels"):
                raise ValueError(f"early_stop_metric {m} requires use_labels.")

    rv = list(out.get("recall_values") or [1, 5, 10, 20])
    need_k = recall_values_needed_for_metrics(out["early_stop_metrics"])
    merged_rv = sorted(set(rv) | set(need_k))
    if merged_rv != rv:
        out["recall_values"] = merged_rv

    # Validate phased early stop
    if out.get("phased_early_stop"):
        es = out["early_stop_metrics"]
        has_ece = any(m.startswith("ece_recall_") for m in es)
        if not has_ece:
            raise ValueError(
                "phased_early_stop requires early_stop_metric to include at least one "
                "ece_recall_* metric (e.g. ece_recall,ece_recall_05,ece_recall_10)."
            )
        # Auto-add recall as Phase 1 metric if not already present.
        if "recall" not in es:
            out["early_stop_metrics"] = ["recall"] + es
            logger.info("phased_early_stop: auto-added 'recall' to early_stop_metrics for Phase 1.")

    # Device auto-detection
    requested_device = str(out.get("device", "auto")).lower()
    out["device"] = "cuda" if requested_device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"

    return out


def _resolve_log_dir(logs_folder: Optional[str],
                     resume_checkpoint: Optional[str] = None,
                     resume_model: Optional[str] = None) -> Optional[Path]:
    """Determine where log files should go based on resume mode and script name."""
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint)
        if resume_path.exists():
            timestamp = datetime.now().strftime("resume_train_%Y-%m-%d_%H-%M-%S")
            return resume_path.parent / timestamp

    elif resume_model:
        resume_path = Path(resume_model)
        if resume_path.exists():
            if "train" in Path(sys.argv[0]).name:
                timestamp = datetime.now().strftime("resume_model_%Y-%m-%d_%H-%M-%S")
                return resume_path.parent / timestamp
            else:  # eval: fixed folder per checkpoint
                eval_dir = resume_path.parent / "eval" / resume_path.stem
                return eval_dir

    if logs_folder:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return Path(logs_folder) / timestamp

    return None


def setup_logging(logs_folder: Optional[str], dry_run: bool = False,
                  resume_checkpoint: Optional[str] = None, resume_model: Optional[str] = None,
                  use_wandb: bool = False):
    """Configure unified logging: console (INFO), info.log, debug.log.
    When use_wandb is True, create log_dir even in dry_run so W&B and eval paths (eval/dataset_name) work."""
    handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", "%H:%M:%S"))
    handlers.append(console_handler)

    log_dir = _resolve_log_dir(logs_folder, resume_checkpoint, resume_model)

    if log_dir and (not dry_run or use_wandb):
        log_dir.mkdir(parents=True, exist_ok=True)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        for filename, level in [("debug.log", logging.DEBUG), ("info.log", logging.INFO)]:
            fh = logging.FileHandler(log_dir / filename, mode="a", encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(file_formatter)
            handlers.append(fh)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers, force=True)

    def exception_handler(type_, value, tb):
        logging.error("Uncaught exception occurred:", exc_info=(type_, value, tb))
    sys.excepthook = exception_handler

    return log_dir


def select_entries(entries: List[Dict[str, Any]], datasets: Any) -> List[Dict[str, Any]]:
    """
    Filter entries by 'datasets' selection.
    datasets can be:
      - "all" or None => no filtering
      - list of names => filter by entry["name"]
    """
    if datasets is None or (isinstance(datasets, str) and datasets.lower() == "all"):
        return entries

    if isinstance(datasets, list):
        wanted = {d.lower() for d in datasets}
        return [e for e in entries if str(e.get("name", "")).lower() in wanted]

    # fallback: no filtering
    return entries


def _log_config_summary(cfg: Dict[str, Any], entries: List[Dict[str, Any]], cfg_path: Path):
    """Log a summary of the resolved configuration."""
    logger.debug(f"Config file: {cfg_path}")
    logger.debug(f"data_folder: {cfg['data_folder']}")
    logger.debug(f"colab: {cfg['colab']}, dry_run: {cfg['dry_run']}")
    if cfg["colab"]:
        logger.debug(f"local_data_folder: {cfg['local_data_folder']}")
    logger.debug(f"entries to process: {[e.get('name') for e in entries]}")
    logger.debug(f"Using device: {cfg['device']}")
    logger.debug(f"method: {cfg.get('method')}, backbone: {cfg.get('backbone')}, "
                f"descriptors_dimension: {cfg.get('descriptors_dimension')}")
    if cfg.get('image_size') is not None:
        logger.debug(f"image_size: {cfg['image_size']}")
    if cfg.get("resume_train") is not None:
        logger.info(f"resume_train: {cfg['resume_train']}")


def build_config():
    """Parse CLI + JSON config, set up logging, validate, and return merged config + entries."""
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    merged = merge_cfg_with_cli(cfg, args)
    merged = normalize(merged)

    log_dir = setup_logging(merged.get("logs_folder"),
                            dry_run=merged.get("dry_run", False),
                            resume_checkpoint=merged.get("resume_train"),
                            resume_model=merged.get("resume_model"),
                            use_wandb=merged.get("use_wandb", False))
    merged['log_dir'] = str(log_dir) if log_dir else None

    for field in ("data_folder", "method"):
        if field not in merged:
            logger.critical(f"Missing required field: '{field}'")
            raise ValueError(f"Missing required field: '{field}'")

    entries = merged.get("entries")
    if not isinstance(entries, list) or len(entries) == 0:
        logger.error("Config must include non-empty list field: 'entries'")
        raise ValueError("Config must include non-empty list field: 'entries'")

    entries = select_entries(entries, merged.get("datasets", None))

    if merged.get("save_config") and log_dir and not merged.get("dry_run"):
        outp = (log_dir / "merged_config.json").expanduser()
        outp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        logger.info(f"Saved merged config to {outp}")

    _log_config_summary(merged, entries, cfg_path)
    return merged, entries


if __name__ == "__main__":
    build_config()