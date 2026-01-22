import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import faiss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

from data.test_dataset import TestDataset

import logging
from configs.parser import build_config
from data.upload_dataset import upload_dataset
from models.get_model import get_model
from utils import visualizations

# Define the logger for this module
# It will inherit the configuration set in setup_logging within parser.py
logger = logging.getLogger(__name__)

def init(args):
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(
        f"Testing with {args['method']} with a {args['backbone']} backbone and descriptors dimension {args['descriptors_dimension']}"
    )
    logger.info(f"The outputs are being saved in {args['log_dir']}")

    model = get_model(args['method'], args['backbone'], args['descriptors_dimension'], args.get('resume_model'))
    device = torch.device(args["device"])
    return device, model

def eval_dataset(args, model, device, dataset_name, eval_ds_path):
    """
    Evaluates the model on a single dataset.
    Saves heavy outputs (descriptors, images) in a dataset-specific subfolder.
    Logs all numerical results (Recalls) to the central log file.
    """
    model = model.eval().to(device)

    # Path for dataset-specific outputs (images, descriptors)
    dataset_output_dir = Path(args['log_dir']) / dataset_name
    if not args['dry_run']:
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
    # Setup Dataset paths
    database_folder = f"{eval_ds_path}/database"
    queries_folder = f"{eval_ds_path}/queries"

    test_ds = TestDataset(
        database_folder,
        queries_folder,
        positive_dist_threshold=args['positive_dist_threshold'],
        image_size=args.get('image_size'),
        use_labels=args['use_labels'],
    )
    logger.info(f"{'='*30}")
    logger.info(f"Testing on {test_ds}")

    with torch.inference_mode():
        logger.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args['num_workers'], batch_size=args['infer_batch_size'], pin_memory=(device == "cuda")
        )
        all_descriptors = np.empty((len(test_ds), args['descriptors_dimension']), dtype="float32")
        for images, indices in tqdm(database_dataloader):
            descriptors, vars = model(images.to(device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
            if args["dry_run"]:
                break

        logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args['num_workers'], batch_size=1, pin_memory=(device == "cuda"))
        for images, indices in tqdm(queries_dataloader):
            descriptors, vars = model(images.to(device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
            if args["dry_run"]:
                break

    queries_descriptors = all_descriptors[test_ds.num_database :]
    database_descriptors = all_descriptors[: test_ds.num_database]

    # Save heavy .npy files in the sub-folder
    if args['save_descriptors'] and not args['dry_run']:
        logger.info(f"Saving the descriptors in {dataset_output_dir}")
        np.save(f"{dataset_output_dir}/queries_descriptors.npy", queries_descriptors)
        np.save(f"{dataset_output_dir}/database_descriptors.npy", database_descriptors)

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args['descriptors_dimension'])
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logger.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(args['recall_values']))

    recalls = None
    recalls_str = ""
    # For each query, check if the predictions are correct
    if args['use_labels']:
        positives_per_query = test_ds.get_positives()
        recalls = np.zeros(len(args['recall_values']))
        for query_index, preds in enumerate(predictions):
            for i, n in enumerate(args['recall_values']):
                if np.any(np.isin(preds[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break

        # Divide by num_queries and multiply by 100, so the recalls are in percentages
        recalls = recalls / test_ds.num_queries * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args['recall_values'], recalls)])
        #logger.info(recalls_str)
        
        # Save a small text file as a backup in the sub-folder
        if not args['dry_run'] and args['datasets_type'] == 'test':
            (dataset_output_dir / "recalls.txt").write_text(recalls_str)

    if args['dry_run']:
        logger.info("Dry run, not saving predictions visualizations.")
        return recalls, recalls_str
    
    # Save visualizations of predictions
    if args['num_preds_to_save'] != 0 and not args['dry_run']:
            logger.info(f"Saving prediction images for {dataset_name} in {dataset_output_dir}")
            visualizations.save_preds(
                predictions[:, : args['num_preds_to_save']], 
                test_ds, 
                str(dataset_output_dir), 
                args['save_only_wrong_preds'], 
                args['use_labels'], 
                args['num_queries_to_save']
            )
    return recalls, recalls_str

if __name__ == "__main__":
    # ---- Load and build config ----       
    cfg, entries = build_config()
    # Upload/Prepare datasets (returns a dict with paths)
    datasetsts_dir = upload_dataset(cfg, entries)
    # Initialize model and device
    device, model = init(cfg)
    # Loop through each dataset entry and evaluate
    for e in entries:
        logger.info(f"Evaluating dataset: {e['name']}")
        recalls, recalls_str = eval_dataset(cfg, model, device, e["name"], datasetsts_dir[e["name"]]['test'])
        logger.info(recalls_str)
    logger.info("="*30)
    logger.info("All evaluations completed successfully.")