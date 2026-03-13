import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import wandb
import torch

from configs.parser import build_config
from data.upload_dataset import upload_dataset
from models.get_model import get_model

logger = logging.getLogger(__name__)


def build_config_and_datasets() -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Path]]]:
    """Shared helper for entrypoints (train.py, eval.py, etc.).
    Parses CLI + config, prepares datasets, returns (cfg, entries, datasets_paths).
    """
    cfg, entries = build_config()
    logger.info(" ".join(sys.argv))
    logger.info(f"The outputs are being saved in {cfg['log_dir']}")
    datasets_paths = upload_dataset(cfg, entries)
    logger.info(f"Loaded {len(entries)} config entr{'y' if len(entries) == 1 else 'ies'}.")
    return cfg, entries, datasets_paths


def init_model(args: Dict[str, Any]):
    """Create model from config and move to device."""
    model = get_model(args)
    device = torch.device(args["device"])
    model = model.to(device)
    return device, model


def _default_wandb_run_name(args: Dict[str, Any], job_type: str) -> str:
    """Build default run name from job_type, var_head, losses and optional flags."""
    losses = args.get("losses") or "ce"
    if isinstance(losses, list):
        losses_str = "_".join(str(l).strip() for l in losses) if losses else "ce"
    else:
        losses_str = str(losses).replace(",", "_").replace(" ", "").strip() or "ce"
    
    parts = [job_type]
    
    # var_head (only if uncertainty mode is enabled)
    if args.get("model_mode") == "uncertainty":
        vht = args.get("var_head_type", "linear")
        parts.append(vht)
    
    # loss
    parts.append(losses_str)
    
    # if init_var flag on
    if args.get("var_init"):
        parts.append("init_var")
        
    if args.get("load_classifiers"):
        parts.append("load_clf")
    if args.get("freeze_model"):
        parts.append("freeze")
    if args.get("resume_model"):
        parts.append("resume_model")
    if args.get("resume_train"):
        parts.append("resume_train")
    return "_".join(parts)


def init_wandb(args: Dict[str, Any], job_type: str = "train") -> bool:
    """Initialize Weights & Biases run if use_wandb is enabled. Returns True if initialized."""
    if not args.get("use_wandb"):
        return False
    run_name = args.get("wandb_run_name") or _default_wandb_run_name(args, job_type)
    if isinstance(run_name, Path):
        run_name = run_name.name
    wandb.init(
        project=args.get("wandb_project", "UncertaintyAwareVPR"),
        name=run_name,
        config=dict(args),
        job_type=job_type,
    )
    return True


def log_wandb(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to W&B if a run is active. No-op otherwise."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def log_wandb_images(images: Dict[str, Path], step: Optional[int] = None) -> None:
    """Log image files to W&B (e.g. plots). Keys become metric names. Skips missing files."""
    if not images or wandb.run is None:
        return
    to_log = {}
    for key, path in images.items():
        p = Path(path)
        if p.exists():
            to_log[key] = wandb.Image(str(p))
    if to_log:
        wandb.log(to_log, step=step)


def save_wandb_logs(log_dir: Optional[str]) -> None:
    """Upload debug.log and info.log from log_dir to the current W&B run. Call before wandb.finish()."""
    if wandb.run is None or not log_dir:
        return
    log_path = Path(log_dir)
    for name in ("debug.log", "info.log"):
        p = log_path / name
        if p.exists():
            wandb.save(str(p), base_path=str(log_path), policy="end")
