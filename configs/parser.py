from __future__ import annotations
from pathlib import Path
import argparse
import json
from typing import Any, Dict, Tuple, Optional, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset preparation: config-first, CLI overrides.")
    p.add_argument("--config", type=str, default="configs/datasets.json", help="Path to datasets config JSON")

    p.add_argument("--data_folder", type=str, default=None, help="Root folder containing raw datasets (overrides config value).")
    p.add_argument("--local_data_folder", type=str, default=None,  help="colab use-Target folder for prepared datasets (overrides config value).")

    # Optional: filter which entries to run 
    p.add_argument("--datasets", type=str, default=None, help='Datasets to process (e.g. "all", "sf_xl", or "sf_xl,pitts30k")')

    # IMPORTANT: tri-state booleans so config merging works:
    # - if not provided => False
    # - if provided => True
    p.add_argument("--colab", action="store_true", help="Run in Google Colab mode (overrides config).")
    p.add_argument("--dry_run", action="store_true", help="Print actions without performing file operations.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    # Optional: write the merged config out
    p.add_argument("--save_config", type=str, default=None, help="Save merged configuration to this path (json)")
    return p.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    path = path.expanduser()
    if not path.exists():
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
    for k in ("data_folder", "local_data_folder"):
        if k in out and out[k] is not None:
            out[k] = str(Path(out[k]).expanduser())
    # datasets: "all" or comma-separated string
    if "datasets" in out and out["datasets"] is not None:
        v = str(out["datasets"]).strip()
        if v.lower() == "all":
            out["datasets"] = "all"
        else:
            out["datasets"] = [s.strip() for s in v.split(",") if s.strip()]
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

    # REQUIRED config fields
    if "data_folder" not in merged:
        raise ValueError("Missing required fields: 'data_folder' (in config or via CLI).")

    entries = merged.get("entries")
    if not isinstance(entries, list) or len(entries) == 0:
        raise ValueError("Config must include non-empty list field: 'entries'")

    # Optional filtering
    entries = select_entries(entries, merged.get("datasets", None))

    # Optional save merged config
    if save_path:
        outp = Path(save_path).expanduser()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(f"Saved merged config to {outp}")

    # Debug prints
    if merged.get("verbose", False):
        print("Config file:", str(cfg_path))
        print("data_folder:", merged["data_folder"])
        print("colab:", merged["colab"], "dry_run:", merged["dry_run"])
        if merged["colab"]:
            print("local_data_folder:", merged["local_data_folder"])
        print("entries:", [e.get("name") for e in entries])

    return merged, entries

if __name__ == "__main__":
    build_config()
