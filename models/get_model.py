import logging
import torch
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
        # Load model weights only when --load_model_weights is set.
        if args.get('load_model_weights'):
            logger.info(f"Loading model weights from {args['resume_model']}")
            checkpoint = torch.load(args['resume_model'], map_location='cpu', weights_only=False)
            if "model_state_dict" in checkpoint:
                model_state_dict = checkpoint["model_state_dict"]
            else:
                model_state_dict = checkpoint
            missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
            if missing:
                logger.debug("State dict missing keys (left at default init): %s", missing)
            if unexpected:
                logger.debug("State dict unexpected keys (ignored): %s", unexpected)
        else:
            logger.info("Skipping model weights (--load_model_weights not set). Model uses default initialization.")
        if args.get('load_classifiers'):
            logger.info("Classifiers will be loaded from the same checkpoint in the training loop.")
    elif args['method'] != "cosplace_pretrained":
        logger.info("No --resume_model provided. Initializing model with default weights (ImageNet).")
    return model