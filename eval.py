# import parser
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import faiss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

# import visualizations
from utils.test_dataset import TestDataset

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
    logger.info(f"The outputs are being saved in {args['logs_folder']}")

    model = get_model(args['method'], args['backbone'], args['descriptors_dimension'], args.get('resume_model'))
    device = torch.device(args["device"])
    model = model.eval().to(device)
    return device, model

def eval(args, model, device, dataset_name):
    database_folder = f"{datasetsts_dir[dataset_name]['test']}/database"
    queries_folder = f"{datasetsts_dir[dataset_name]['test']}/queries"

    test_ds = TestDataset(
        database_folder,
        queries_folder,
        positive_dist_threshold=args['positive_dist_threshold'],
        image_size=args.get('image_size'),
        use_labels=args['use_labels'],
    )
    logger.info(f"Testing on {test_ds}")

    with torch.inference_mode():
        logger.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args['num_workers'], batch_size=args['batch_size']
        )
        all_descriptors = np.empty((len(test_ds), args['descriptors_dimension']), dtype="float32")
        for images, indices in tqdm(database_dataloader):
            descriptors = model(images.to(device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args['num_workers'], batch_size=1)
        for images, indices in tqdm(queries_dataloader):
            descriptors = model(images.to(device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

    queries_descriptors = all_descriptors[test_ds.num_database :]
    database_descriptors = all_descriptors[: test_ds.num_database]

    if args['save_descriptors']:
        logger.info(f"Saving the descriptors in {args['logs_folder']}")
        np.save(f"{args['logs_folder']}/queries_descriptors.npy", queries_descriptors)
        np.save(f"{args['logs_folder']}/database_descriptors.npy", database_descriptors)

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args['descriptors_dimension'])
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logger.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(args['recall_values']))

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
        logger.info(recalls_str)

    # Save visualizations of predictions
    if args['num_preds_to_save'] != 0:
        logger.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(
            predictions[:, : args['num_preds_to_save']], test_ds, args['logs_folder'], args['save_only_wrong_preds'], args['use_labels'], args['num_queries_to_save']
        )
    return recalls, recalls_str

if __name__ == "__main__":
    # ---- Load and build config ----       
    cfg, entries = build_config()
    datasetsts_dir = upload_dataset(cfg, entries)
    device, model = init(cfg)
    for e in entries:
        logger.info(f"Evaluating dataset: {e['name']}")
        eval(cfg, model, device, e["name"])
    logger.info("Evaluation completed.")
