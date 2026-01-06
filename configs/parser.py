from __future__ import annotations
from pathlib import Path
import argparse
import json
import logging
from typing import Any, Dict, Tuple, Optional, List

# Create a logger for this module
logger = logging.getLogger(__name__)

def setup_logging(logs_folder: Optional[str], verbose: bool):
    """
    Configures logging to both the console and a file.
    If logs_folder is provided, a file named 'dataset_prep.log' will be created there.
    """
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]  # Console handler

    if logs_folder:
        log_dir = Path(logs_folder)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "dataset_prep.log"
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )

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

    # Optional: write the merged config out
    p.add_argument("--save_config", type=str, default=None, help="Save merged configuration to this path (json)")

    # model parameters
    p.add_argument("--backbone", type=str, default="ResNet18", help="basic backbone model")
    p.add_argument("--fc_output_dim", type=int, default=512, help="dimension of the output feature vector")

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
    save_config = cli.pop("save_config", None)

    overrides = {k: v for k, v in cli.items() if v is not None}
    merged = dict(cfg)
    merged.update(overrides)
    return merged, save_config


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

    merged, save_path = merge_cfg_with_cli(cfg, args)
    merged = normalize(merged)

    # Initialize the logging system
    setup_logging(merged.get("logs_folder"), merged.get("verbose", False))

    # REQUIRED config fields
    if "data_folder" not in merged:
        logger.critical("Missing required fields: 'data_folder'")
        raise ValueError("Missing required fields: 'data_folder' (in config or via CLI).")

    entries = merged.get("entries")
    if not isinstance(entries, list) or len(entries) == 0:
        logger.error("Config must include non-empty list field: 'entries'")
        raise ValueError("Config must include non-empty list field: 'entries'")

    # Optional filtering
    entries = select_entries(entries, merged.get("datasets", None))

    # Optional save merged config
    if save_path:
        outp = Path(save_path).expanduser()
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

    return merged, entries

if __name__ == "__main__":
    build_config()