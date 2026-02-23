import logging
from typing import Any, Dict

import torch

from models.model_mode import GeneralModelWrapper, deliver_model

logger = logging.getLogger(__name__)


def _load_weights(model: torch.nn.Module, checkpoint_path: str):
    """Load model weights from a checkpoint file (strict=False to allow mismatches)."""
    logger.info(f"Loading model weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.debug("State dict missing keys (left at default init): %s", missing)
    if unexpected:
        logger.debug("State dict unexpected keys (ignored): %s", unexpected)


def get_model(args: Dict[str, Any]) -> torch.nn.Module:
    """Build model and optionally load pretrained weights."""
    method = args["method"]

    if method == "cosplace":
        model = deliver_model(args)
    elif method == "cosplace_pretrained":
        logger.info(f"Loading pretrained model from torch.hub: backbone={args['backbone']}, dim={args['descriptors_dimension']}")
        model = torch.hub.load("gmberton/cosplace", "get_trained_model", args["backbone"], args["descriptors_dimension"])
        model = GeneralModelWrapper(model)
    else:
        raise ValueError(f"Unknown method: {method}")

    resume_path = args.get("resume_model")
    if resume_path is not None:
        if args.get("load_model_weights"):
            _load_weights(model, resume_path)
        else:
            logger.info("Skipping model weights (--load_model_weights not set).")
    elif method != "cosplace_pretrained":
        logger.info("No --resume_model provided. Model uses default weights (ImageNet).")

    return model
