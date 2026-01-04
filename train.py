
from configs.parser import build_config
from data.upload_dataset import upload_dataset

if __name__ == "__main__":
    # ---- Load and build config ----       
    cfg, entries = build_config()
    datasetsts_dir = upload_dataset(cfg, entries)