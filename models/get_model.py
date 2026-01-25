import torch
import torch.nn as nn
from models.cosplace_uncertainty.cosplace_model import cosplace_network
import logging

from models.model_mode import deliver_model, GeneralModelWrapper

logger = logging.getLogger(__name__)

#def get_model(method, backbone=None, descriptors_dimension=None, resume_model=None, train_all_layers=False):
def get_model(args):
    if args['method'] == "cosplace":
        # model = cosplace_network.GeoLocalizationNet(backbone, descriptors_dimension, train_all_layers)
        model = deliver_model(args)
    elif args['method'] == "cosplace_pretrained":
        logger.info(f"Loading pretrained model from torch.hub: backbone={args['backbone']}, dim={args['descriptors_dimension']}")
        model = torch.hub.load("gmberton/cosplace", "get_trained_model", args['backbone'], args['descriptors_dimension'])
        model = GeneralModelWrapper(model)
    if args.get('resume_model') is not None:
        logger.info(f"Loading model from {args['resume_model']}")
        model_state_dict = torch.load(args['resume_model'], map_location='cpu')
        model.load_state_dict(model_state_dict)
    elif args['method'] != "cosplace_pretrained":
        logger.info("No --resume_model provided. Initializing model with default weights (ImageNet).")
    return model