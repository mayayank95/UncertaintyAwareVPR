from __future__ import annotations
from pathlib import Path
import argparse
import json
import logging
from datetime import datetime
from typing import Any, Dict, Tuple, Optional, List
import sys
import torch

from models.get_model import get_model

# Create a logger for this module
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset preparation: config-first, CLI overrides.")
    p.add_argument("--config", type=str, default="configs/datasets.json", help="Path to datasets config JSON")

    p.add_argument("--data_folder", type=str, default=None, help="Root folder containing raw datasets (overrides config value).")
    p.add_argument("--local_data_folder", type=str, default=None,  help="colab use-Target folder for prepared datasets (overrides config value).")
    p.add_argument("--logs_folder", type=str, default=None,  help="folder to save logs (overrides config value).")

    # Optional: filter which entries to run 
    p.add_argument("--datasets", type=str, default=None, help='Datasets to process (e.g. "all", "sf_xl", or "sf_xl,pitts30k")')
    p.add_argument("--datasets_type", type=str, default="all", help='Datasets type to upload(e.g. "all", "train", or "test", "val")')

    # Resume parameters
    p.add_argument("--resume_train", type=str, default=None,
                        help="path to checkpoint to resume, e.g. logs/.../last_checkpoint.pth")
    p.add_argument("--resume_model", type=str, default=None,
                        help="path to model to resume, e.g. logs/.../best_model.pth")
                        
    # IMPORTANT: tri-state booleans so config merging works:
    # - if not provided => False
    # - if provided => True
    p.add_argument("--colab", action="store_true", help="Run in Google Colab mode (overrides config).")
    p.add_argument("--dry_run", action="store_true", help="Print actions without performing file operations.")

    # Optional: save post-merge config
    p.add_argument("--save_config", action="store_true", help="Save merged configuration to logs folder")
    p.add_argument("--save_descriptors",action="store_true", help="set to True if you want to save the descriptors extracted by the model")
    
    # model parameters
    p.add_argument("--backbone", type=str, default=None, help="basic backbone model")
    p.add_argument("--descriptors_dimension", type=int, default=None, help="dimension of the output feature vector")
    p.add_argument("--method", type=str, default=None, help="model name")
    p.add_argument("--positive_dist_threshold", type=int, default=None, help="Distance in meters for a prediction to be considered a positive.")
    p.add_argument("--image_size", type=int, default=None, help="Resize images to this size (square).")
    p.add_argument("--use_labels", action="store_true", help="Use UTM coordinates from image paths for evaluation.") 
    p.add_argument("--train_all_layers", action="store_true", help="If true, train all layers of the backbone")
    p.add_argument("--resize_test_imgs", action="store_true", help="If the test images should be resized to image_size along the shorter side while maintaining aspect ratio")
    

    # system parameters
    p.add_argument("--device", type=str, default="auto", help="Device to use: 'cuda', 'cpu', or 'auto'")
    p.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers")
    p.add_argument("--use_amp16", action="store_true", help="use Automatic Mixed Precision")
    
    # evaluation parameters
    p.add_argument("--recall_values", type=int, nargs="+", default=[1, 5, 10, 20], help="Recall values to compute during evaluation.")
    p.add_argument("--infer_batch_size", type=int, default=16, help="Batch size for inference (validating and testing)")
  
    # visualization parameters
    p.add_argument("--num_preds_to_save", type=int, default=3, help="Number of predictions to save per query.")
    p.add_argument("--num_queries_to_save", type=int, default=3, help="Number of queries to save their predictions.")
    p.add_argument("--save_only_wrong_preds", action="store_true", help="If set, only save wrongly predicted queries.") 

    # training parameters
    p.add_argument("--cudnn_benchmark", action="store_true", 
                    help="Set torch.backends.cudnn.benchmark to True. Faster, but non-deterministic.")
    p.add_argument("--lr", type=float, default=0.00001, help="_")
    p.add_argument("--seed", type=int, default=0, help="_")
    p.add_argument("--classifiers_lr", type=float, default=0.01, help="_")
    p.add_argument("--batch_size", type=int, default=None, help="Batch size for DataLoader.")
    p.add_argument("--iterations_per_epoch", type=int, default=10000, help="_")
    p.add_argument("--epochs_num", type=int, default=50, help="_")
    p.add_argument("--patience", type=int, default=5, help="Patience for early stopping (epochs without improvement)")

    # Data augmentation
    p.add_argument("--augmentation_device", type=str, default="cuda", choices=["cuda", "cpu"], help="on which device to run data augmentation")
    p.add_argument("--brightness", type=float, default=0.7, help="_")
    p.add_argument("--contrast", type=float, default=0.7, help="_")
    p.add_argument("--hue", type=float, default=0.5, help="_")
    p.add_argument("--saturation", type=float, default=0.7, help="_")
    p.add_argument("--random_resized_crop", type=float, default=0.5, help="_")

    # CosPlace Groups parameters
    p.add_argument("--M", type=int, default=10, help="_")
    p.add_argument("--alpha", type=int, default=30, help="_")
    p.add_argument("--N", type=int, default=5, help="_")
    p.add_argument("--L", type=int, default=2, help="_")
    p.add_argument("--groups_num", type=int, default=8, help="_")
    p.add_argument("--min_images_per_class", type=int, default=10, help="_")

    # uncertainty parameters
    p.add_argument("--model_mode", type=str, default=None, help="model mode: basic/uncertainty")
    p.add_argument("--sigma_dim", type=int, default=None, help="dimension of the output uncertainty vector (variance)")
    p.add_argument("--uncertainty_lambda", type=float, default=1.0, help="Weight for the uncertainty loss in uncertainty mode")
    p.add_argument("--uncertainty_loss", type=str, default=None, help="Uncertainty loss type: gaussian_nll or gaussian_cosine")
    p.add_argument("--use_variance_linear", action="store_true", help="If set, adds a linear layer before the softplus in the variance head")
    p.add_argument("--separate_variance_aggregation", action="store_true", help="If set, use a separate aggregation module (copy of mean) for variance calculation.")
    p.add_argument("--uncertainty_max_queries", type=int, default=30, help="Max number of queries to use for uncertainty correlation calculation.")
    return p.parse_args()  
    

def load_config(path: Path) -> Dict[str, Any]:
    path = path.expanduser()
    if not path.exists():
        logger.error(f"Config not found: {path}") # Log error before raising
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def merge_cfg_with_cli(cfg: Dict[str, Any], args: argparse.Namespace) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Merge rule: merged = cfg + (CLI overrides for keys where CLI value is not None).
    """
    cli = vars(args).copy()
    cli.pop("config", None)

    overrides = {k: v for k, v in cli.items() if v is not None}
    merged = dict(cfg)
    merged.update(overrides)
    return merged


def normalize(merged: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize common fields to expected types/format.
    """
    out = dict(merged)

    # Paths: store as strings in config, but normalize to expanded string paths
    for k in ("data_folder", "local_data_folder", "logs_folder", "resume_train", "resume_model"):
        if k in out and out[k] is not None:
            out[k] = str(Path(out[k]).expanduser())

    # datasets: "all" or comma-separated string
    for element in ["datasets", "datasets_type"]:
        if element in out and out[element] is not None:
            v = str(out[element]).strip()
            if v.lower() == "all":
                out[element] = "all"
            else:
                out[element] = [s.strip() for s in v.split(",") if s.strip()]

    # Device auto-detection
    requested_device = str(out.get("device", "auto")).lower()
    if requested_device == "auto" or requested_device == "cuda":
        out["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        out["device"] = "cpu"
  
    return out


def setup_logging(logs_folder: Optional[str], dry_run: bool = False, resume_checkpoint: Optional[str] = None, resume_model: Optional[str] = None):
    """
    Configures a unified logging system:
    - Console: Shows clean INFO messages.
    - info.log: Stores clean INFO level logs and above.
    - debug.log: Stores detailed DEBUG logs (the full technical record).
    """
    
    # Root level acts as the master gatekeeper
    root_level = logging.DEBUG
    
    handlers = []

    # 1. Console Handler - Keep the terminal output clean and readable
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  
    # Adding short timestamp to console for real-time tracking
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", "%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    log_dir = None
    
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint)
        if resume_path.exists():
            original_log_dir = resume_path.parent
            timestamp = datetime.now().strftime("resume_%Y-%m-%d_%H-%M-%S")
            log_dir = original_log_dir / timestamp
    elif resume_model:
        resume_path = Path(resume_model)
        if resume_path.exists():
            original_log_dir = resume_path.parent
            if "train" in Path(sys.argv[0]).name:
                timestamp = datetime.now().strftime("resume_model_%Y-%m-%d_%H-%M-%S")
            else:
                timestamp = datetime.now().strftime("eval_%Y-%m-%d_%H-%M-%S")
            log_dir = original_log_dir / timestamp

    if log_dir is None and logs_folder:
        # Create a unique timestamped folder for this specific run
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = Path(logs_folder) / timestamp

    if log_dir and not dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)

        # Detailed formatter for files (includes full date and time)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 2. Debug File Handler - Captures everything (The "Black Box" of the run)
        debug_file = log_dir / "debug.log"
        debug_handler = logging.FileHandler(debug_file, encoding="utf-8")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(file_formatter)
        handlers.append(debug_handler)

        # 3. Info File Handler - Captures only high-level progress
        info_file = log_dir / "info.log"
        info_handler = logging.FileHandler(info_file, encoding="utf-8")
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(file_formatter)
        handlers.append(info_handler)

    # Apply configuration to the global logging system
    logging.basicConfig(
        level=root_level,
        handlers=handlers,
        force=True  # Overrides any existing logging configuration
    )

    # Exception Hook
    # Ensures that if the script crashes, the traceback is captured in the log files
    def exception_handler(type_, value, tb):
        logging.error("Uncaught exception occurred:", exc_info=(type_, value, tb))
        logging.info("Execution finished with errors.")
    
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


def build_config():
    args = parse_args() # Load CLI args

    # Load config from file
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    merged = merge_cfg_with_cli(cfg, args)
    merged = normalize(merged)

    # Initialize the logging system
    log_dir = setup_logging(merged.get("logs_folder"), 
                            dry_run=merged.get("dry_run", False), 
                            resume_checkpoint=merged.get("resume_train"),
                            resume_model=merged.get("resume_model"))
    merged['log_dir'] = str(log_dir) if log_dir else None # Ensure logs_folder is set to the actual log_dir used

    # REQUIRED config fields
    required_fields = ["data_folder", "method"]
    for field in required_fields:
        if field not in merged:
            logger.critical(f"Missing required field: '{field}'")
            raise ValueError(f"Missing required field: '{field}'")

     # Validate entries
    entries = merged.get("entries")
    if not isinstance(entries, list) or len(entries) == 0:
        logger.error("Config must include non-empty list field: 'entries'")
        raise ValueError("Config must include non-empty list field: 'entries'")

    # Optional filtering
    entries = select_entries(entries, merged.get("datasets", None))

    # Optional save merged config
    if merged.get("save_config") and log_dir and not merged.get("dry_run"):
        outp = log_dir / "merged_config.json"
        # Save merged config to specified path  
        outp = Path(outp).expanduser()  
        outp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        logger.info(f"Saved merged config to {outp}")

    # Debug logs (replaces prints)
    logger.debug(f"Config file: {cfg_path}")
    logger.debug(f"data_folder: {merged['data_folder']}")
    logger.debug(f"colab: {merged['colab']}, dry_run: {merged['dry_run']}")
    if merged["colab"]:
        logger.debug(f"local_data_folder: {merged['local_data_folder']}")
    logger.info(f"entries to process: {[e.get('name') for e in entries]}")
    logger.info(f"Using device: {merged['device']}")
    logger.info(f"method: {merged.get('method')}, backbone: {merged.get('backbone')}, descriptors_dimension: {merged.get('descriptors_dimension')}")    
    if merged.get('image_size') is not None:
        logger.info(f"image_size: {merged.get('image_size')}")
    if merged.get("resume_train") is not None:
        logger.info(f"resume_train: {merged.get('resume_train')}")

    return merged, entries

def init_model(args):
    logger.info(" ".join(sys.argv))
    logger.info(f"The outputs are being saved in {args['log_dir']}")

    model = get_model(args)
    device = torch.device(args["device"])
    model = model.to(device)
    return device, model

if __name__ == "__main__":
    build_config()