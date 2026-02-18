import logging
from typing import Any, Dict, List, Tuple

from configs.parser import build_config
from data.upload_dataset import upload_dataset


def build_config_and_datasets() -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Shared helper for entrypoints (train.py, eval.py, etc.).

    - Parses CLI + config via configs.parser.build_config
    - Uploads / prepares datasets via data.upload_dataset.upload_dataset
    - Returns (cfg, entries, datasets_paths)
    """
    logger = logging.getLogger(__name__)

    cfg, entries = build_config()
    datasets_paths = upload_dataset(cfg, entries)

    logger.info(f"Loaded {len(entries)} config entr{'y' if len(entries) == 1 else 'ies'}.")

    return cfg, entries, datasets_paths

