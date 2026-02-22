import torch
import torch.nn as nn
from models.cosplace_uncertainty.cosplace_model import cosplace_network
import logging

from models.model_mode import deliver_model, GeneralModelWrapper

logger = logging.getLogger(__name__)


def _load_backbone_only(model, state_dict, logger):
    """
    Load only backbone.* from state_dict into model. All other layers (aggregation,
    final_l2, var_head, variance_aggregation) keep their default/random initialization.
    """
    backbone_state = {k: v for k, v in state_dict.items() if k.startswith("backbone.")}
    if not backbone_state:
        logger.warning("No backbone keys found in checkpoint; backbone stays at default init.")
        return
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    logger.info(f"Loaded {len(backbone_state)} backbone parameters from checkpoint. "
                "Aggregation, final_l2, and var_head are initialized from scratch.")


def get_model(args):
    if args['method'] == "cosplace":
        model = deliver_model(args)
    elif args['method'] == "cosplace_pretrained":
        logger.info(f"Loading pretrained model from torch.hub: backbone={args['backbone']}, dim={args['descriptors_dimension']}")
        model = torch.hub.load("gmberton/cosplace", "get_trained_model", args['backbone'], args['descriptors_dimension'])
        model = GeneralModelWrapper(model)
    if args.get('resume_model') is not None:
        
        
        logger.info(f"Loading model from {args['resume_model']}")
        checkpoint = torch.load(args['resume_model'], map_location='cpu', weights_only=False)
        if "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
        else:
            model_state_dict = checkpoint
        is_uncertainty_model = args.get('model_mode') == 'uncertainty'
        looks_like_basic = not any(k.startswith("var_head.") for k in model_state_dict)
        if is_uncertainty_model and looks_like_basic:
            # Create uncertainty model; init only backbone from resume, rest from scratch.
            logger.info("Creating uncertainty model: backbone from resume checkpoint, aggregation/final_l2/var_head from scratch.")
            _load_backbone_only(model, model_state_dict, logger)
        else:
            # Same architecture (e.g. uncertainty->uncertainty) or basic->basic: load full state dict.
            missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
            if missing:
                logger.info(f"State dict missing keys: {missing}")
            if unexpected:
                logger.info(f"State dict unexpected keys (ignored): {unexpected}")
        if args.get('load_classifiers'):
            logger.info("Classifiers will be loaded from the same checkpoint in the training loop.")
    elif args['method'] != "cosplace_pretrained":
        logger.info("No --resume_model provided. Initializing model with default weights (ImageNet).")
    return model