# upload.py
from pathlib import Path
import zipfile
import logging
from typing import Dict, List

# Create a logger for this module
logger = logging.getLogger(__name__)

def _ensure_dir(dst):
    # check if path exists else create it
    if not dst.exists():
        dst.mkdir(parents=True, exist_ok=True)

def unzip_folder(src: Path, dst: Path, data_type: List[str]):
    for zip_path in src.glob("*.zip"):
        if zip_path.stem.lower() not in data_type and data_type != "all":
            logger.debug(f"Skipping {zip_path.name} as it is not in the specified datasets_type.")
            continue    
        out_dir = dst / zip_path.stem  # folder named like the zip
        logger.info(f"Unzipping {zip_path.name} -> {out_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                # avoid empty directories
                if member.is_dir(): continue

                # trick: cut the first folder name from the path
                # "folder/sub/file.txt" becomes "sub/file.txt"
                member.filename = str(Path(*Path(member.filename).parts[1:]))
                zf.extract(member, out_dir)

def upload_dataset(cfg, entries) -> Dict[str, Dict[str, Path]]:
    
    colab = cfg["colab"]
    dry_run = cfg["dry_run"]
    data_type = cfg["datasets_type"]

    # ---- Convert string paths (from parser.normalize) to Path objects ONCE ----
    data_root = Path(cfg["data_folder"]).expanduser()
    local_root = Path(cfg["local_data_folder"]).expanduser() if colab else None

    # ---- Announce mode once ----
    if colab:
        logger.info(f"Running in Google Colab mode: materializing datasets into {str(local_root)}")
    else:
        logger.info(f"Running in local mode: using datasets in-place from {str(data_root)}")

    # Collect train/validation/test paths for each dataset
    dataset_paths: Dict[str, Dict[str, Path]] = {}

    for e in entries:
        # ---- Validate entry ----
        name = e.get("name")
        if not name:
            logger.error("Validation failed: Each entry must have a non-empty 'name'")
            raise ValueError("Each entry must have a non-empty 'name'")

        src_rel = e.get("src_rel")
        if not src_rel:
            logger.error(f"Validation failed: Entry '{name}' missing required field: src_rel")
            raise ValueError(f"Entry '{name}' missing required field: src_rel")

        dst_rel = e.get("dst_rel") or name

        # ---- Build src (must exist) ----
        src = (data_root / src_rel).resolve()
        if not src.exists():
            logger.error(f"[{name}] Source not found: {src}")
            raise FileNotFoundError(f"[{name}] Source not found: {src}")

        # ---- Decide dst ----
        if colab:
            dst = (local_root / dst_rel)  # do NOT resolve: may not exist yet
        else:
            dst = src  # local: in-place

        # ---- Local: nothing to do ----
        if not colab:
            logger.debug(f"[{name}] Using dataset in-place at {dst}")
                
        # ---- Colab: ensure destination exists ----
        else: 
            if not dst.exists():
                if dry_run:
                    logger.info(f"[{name}] DRY RUN: would create destination folder: {dst}")
                else:
                    logger.info(f"[{name}] Creating destination folder: {dst}")
                    _ensure_dir(dst)
            if not dry_run:
                logger.info(f"[{name}] Extracting dataset from {src} to {dst}")  
                unzip_folder(src, dst, data_type) # Now it runs whether dst was just created or already existed

        # ---- Record train/validation/test paths (non-strict resolve to allow non-existing in dry-run) ----
        train_p = (dst / 'train').resolve(strict=False)
        val_p = (dst / 'val').resolve(strict=False)
        test_p = (dst / 'test').resolve(strict=False)

        dataset_paths[name] = {
            'train': train_p,
            'validation': val_p,
            'test': test_p,
        }

    return dataset_paths