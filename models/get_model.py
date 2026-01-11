import torch
from models.cosplace_uncertainty.cosplace_model import cosplace_network
import logging

logger = logging.getLogger(__name__)

def get_model(method, backbone=None, descriptors_dimension=None, resume_model=None):
    if method == "cosplace":
        model = cosplace_network.GeoLocalizationNet(backbone, descriptors_dimension)
    elif method == "cosplace_pretrained":
        model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone, descriptors_dimension)
    if resume_model is not None:
        logger.info(f"Loading model from {resume_model}")
        model_state_dict = torch.load(resume_model)
        model.load_state_dict(model_state_dict)
    else:
        logger.info("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                    "Using randomly initialized weights.")
    return model