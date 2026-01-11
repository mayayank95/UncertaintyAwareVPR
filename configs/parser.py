from __future__ import annotations
from pathlib import Path
import argparse
import json
import logging
from datetime import datetime
from typing import Any, Dict, Tuple, Optional, List
import sys
import torch

# Create a logger for this module
logger = logging.getLogger(__name__)

def setup_logging(logs_folder: Optional[str], verbose: bool):
    """
    Configures a unified logging system:
    - Console: Shows clean INFO messages (no clutter).
    - File: Stores detailed DEBUG logs in a timestamped folder.
    """
    # The Root level is the "Master Gatekeeper"
    root_level = logging.DEBUG if verbose else logging.INFO 
    
    # Define handlers list
    handlers = []

    # 1. Console Handler (The "Screen" output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Screen stays clean even in verbose mode
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    if logs_folder:
        # Create a unique folder for this specific run
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = Path(logs_folder) / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. File Handler (The "Record" output)
        log_file = log_dir / "main_execution.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(root_level)  # Saves everything allowed by the master gatekeeper
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Apply configuration to the global logging system
    logging.basicConfig(
        level=root_level,
        handlers=handlers,
        force=True  # Ensures this config overrides any defaults
    )
    return log_dir     

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset preparation: config-first, CLI overrides.")
    p.add_argument("--config", type=str, default="configs/datasets.json", help="Path to datasets config JSON")

    p.add_argument("--data_folder", type=str, default=None, help="Root folder containing raw datasets (overrides config value).")
    p.add_argument("--local_data_folder", type=str, default=None,  help="colab use-Target folder for prepared datasets (overrides config value).")
    p.add_argument("--logs_folder", type=str, default=None,  help="folder to save logs (overrides config value).")

    # Optional: filter which entries to run 
    p.add_argument("--datasets", type=str, default=None, help='Datasets to process (e.g. "all", "sf_xl", or "sf_xl,pitts30k")')
    p.add_argument("--datasets_type", type=str, default="all", help='Datasets type to upload(e.g. "all", "train", or "test", "val")')

    # IMPORTANT: tri-state booleans so config merging works:
    # - if not provided => False
    # - if provided => True
    p.add_argument("--colab", action="store_true", help="Run in Google Colab mode (overrides config).")
    p.add_argument("--dry_run", action="store_true", help="Print actions without performing file operations.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    # Optional: save post-merge config
    p.add_argument("--save_config", action="store_true", help="Save merged configuration to logs folder")
    p.add_argument("--save_descriptors",action="store_true", help="set to True if you want to save the descriptors extracted by the model")
    
    # model parameters
    p.add_argument("--backbone", type=str, default=None, help="basic backbone model")
    p.add_argument("--descriptors_dimension", type=int, default=None, help="dimension of the output feature vector")
    p.add_argument("--resume_model", type=str, default=None, help="model checkpoint to resume training from/evaluate")
    p.add_argument("--method", type=str, default=None, help="model name")
    p.add_argument("--positive_dist_threshold", type=int, default=None, help="Distance in meters for a prediction to be considered a positive.")
    p.add_argument("--image_size", type=int, default=None, help="Resize images to this size (square).")
    p.add_argument("--use_labels", action="store_true", help="Use UTM coordinates from image paths for evaluation.") 
    p.add_argument("--batch_size", type=int, default=None, help="Batch size for DataLoader.")

    # system parameters
    p.add_argument("--device", type=str, default="auto", help="Device to use: 'cuda', 'cpu', or 'auto'")
    p.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers")

    # evaluation parameters
    p.add_argument("--recall_values", type=int, nargs="+", default=[1, 5, 10, 20], help="Recall values to compute during evaluation.")

    # visualization parameters
    p.add_argument("--num_preds_to_save", type=int, default=3, help="Number of predictions to save per query.")
    p.add_argument("--num_queries_to_save", type=int, default=3, help="Number of queries to save their predictions.")
    p.add_argument("--save_only_wrong_preds", action="store_true", help="If set, only save wrongly predicted queries.") 

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
    for k in ("data_folder", "local_data_folder", "logs_folder"): # added logs_folder to normalization
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
    log_dir = setup_logging(merged.get("logs_folder"), merged.get("verbose", False))
    merged['logs_folder'] = str(log_dir)  # Ensure logs_folder is set to the actual log_dir used

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
    if merged.get("save_config"):
        outp = f"{log_dir}/merged_config.json"
        # Save merged config to specified path  
        outp = Path(outp).expanduser()  
        outp.parent.mkdir(parents=True, exist_ok=True)      
        outp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        logger.info(f"Saved merged config to {outp}")

    # Debug logs (replaces prints)
    if merged.get("verbose", False):
        logger.debug(f"Config file: {cfg_path}")
        logger.debug(f"data_folder: {merged['data_folder']}")
        logger.debug(f"colab: {merged['colab']}, dry_run: {merged['dry_run']}")
        if merged["colab"]:
            logger.debug(f"local_data_folder: {merged['local_data_folder']}")
        logger.info(f"entries to process: {[e.get('name') for e in entries]}")
        logger.debug(f"Using device: {merged['device']}")
        logger.info(f"method: {merged.get('method')}, backbone: {merged.get('backbone')}, descriptors_dimension: {merged.get('descriptors_dimension')}")    
        if merged.get('image_size') is not None:
            logger.info(f"image_size: {merged.get('image_size')}")
        if merged.get("resume_model") is not None:
            logger.info(f"resume_model: {merged.get('resume_model')}")
        

    return merged, entries

if __name__ == "__main__":
    build_config()