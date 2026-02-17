import torch
import torch.nn as nn
from models.cosplace_uncertainty.cosplace_model import cosplace_network
import logging

from models.model_mode import deliver_model, GeneralModelWrapper

logger = logging.getLogger(__name__)

def get_model(args):
    if args['method'] == "cosplace":
        model = deliver_model(args)
    elif args['method'] == "cosplace_pretrained":
        logger.info(f"Loading pretrained model from torch.hub: backbone={args['backbone']}, dim={args['descriptors_dimension']}")
        model = torch.hub.load("gmberton/cosplace", "get_trained_model", args['backbone'], args['descriptors_dimension'])
        model = GeneralModelWrapper(model)
    if args.get('resume_model') is not None:
        if args.get('load_classifiers'):
            logger.info("Skipping model weights loading from resume_model because --load_classifiers is set.")
        else:
            logger.info(f"Loading model from {args['resume_model']}")
            checkpoint = torch.load(args['resume_model'], map_location='cpu')
            if "model_state_dict" in checkpoint:
                model_state_dict = checkpoint["model_state_dict"]
            else:
                model_state_dict = checkpoint
            model.load_state_dict(model_state_dict)
    elif args['method'] != "cosplace_pretrained":
        logger.info("No --resume_model provided. Initializing model with default weights (ImageNet).")
    return model