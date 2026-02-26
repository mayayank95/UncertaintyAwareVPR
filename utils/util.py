import logging
import shutil
from pathlib import Path
from typing import Dict, List, Type

import torch
from losses.cosface_loss import MarginCosineProduct


def cosine_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine distance between corresponding vectors. Returns shape [B], range [0, 2]."""
    cos_sim = torch.sum(x1 * x2, dim=-1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    return 1.0 - cos_sim


def move_to_device(optimizer: Type[torch.optim.Optimizer], device: str):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def save_checkpoint(state: dict, is_best: bool, output_folder: str,
                    ckpt_filename: str = "last_checkpoint.pth"):
    # TODO it would be better to move weights to cpu before saving
    checkpoint_path = f"{output_folder}/{ckpt_filename}"
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save({
            "model_state_dict": state["model_state_dict"],
            "classifiers_state_dict": state["classifiers_state_dict"]
        }, f"{output_folder}/best_model.pth")


def resume_train(device: str, args: Dict, output_folder: str, model: torch.nn.Module,
                 model_optimizer: Type[torch.optim.Optimizer], classifiers: List[MarginCosineProduct],
                 classifiers_optimizers: List[Type[torch.optim.Optimizer]]):
    """Load full training state: model, optimizer, classifiers, and epoch counter."""
    logging.info(f"Loading checkpoint: {args['resume_train']}")
    checkpoint = torch.load(args['resume_train'], map_location='cpu', weights_only=False)

    start_epoch_num = checkpoint["epoch_num"]
    
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)
    
    model = model.to(device)
    model_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    assert args['groups_num'] == len(classifiers) == len(classifiers_optimizers) == \
        len(checkpoint["classifiers_state_dict"]) == len(checkpoint["optimizers_state_dict"]), \
        (f"{args['groups_num']}, {len(classifiers)}, {len(classifiers_optimizers)}, "
         f"{len(checkpoint['classifiers_state_dict'])}, {len(checkpoint['optimizers_state_dict'])}")
    
    for c, sd in zip(classifiers, checkpoint["classifiers_state_dict"]):
        # Move classifiers to GPU before loading their optimizers
        c = c.to(device)
        c.load_state_dict(sd)
    for c, sd in zip(classifiers_optimizers, checkpoint["optimizers_state_dict"]):
        # Skip loading optimizer state if it was saved from frozen classifiers (empty optimizers).
        if len(sd.get("state", {})) > 0:
            c.load_state_dict(sd)
    for c in classifiers:
        # Move classifiers back to CPU to save some GPU memory
        c = c.cpu()
    
    best_val_recall1 = checkpoint["best_val_recall1"]
    
    # Copy best model to current output_folder
    best_model_source_path = Path(args['resume_train']).parent / "best_model.pth"
    if best_model_source_path.exists():
        shutil.copy(best_model_source_path, output_folder)
        logging.info(f"Copied best model from previous run to {output_folder}")
    
    return model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num
