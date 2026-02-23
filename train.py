import logging
import multiprocessing
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm

from configs.runtime import build_config_and_datasets, init_model
from data.test_dataset import TestDataset
from data.train_dataset import TrainDataset
from eval import eval_dataset
from losses import cosface_loss
from losses.gaussian_cosine_loss import GaussianCosineLoss
from utils import augmentations, commons, util

logger = logging.getLogger(__name__)


def _load_and_freeze_classifiers(classifiers, checkpoint_path):
    """Load classifier weights from checkpoint and freeze them. Returns updated optimizers (None per classifier)."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or "classifiers_state_dict" not in checkpoint:
        logger.warning("No classifiers_state_dict found in checkpoint.")
        return None
    if len(checkpoint["classifiers_state_dict"]) != len(classifiers):
        logger.warning(f"Skipping classifiers load: checkpoint has {len(checkpoint['classifiers_state_dict'])}, config has {len(classifiers)}.")
        return None
    for c, sd in zip(classifiers, checkpoint["classifiers_state_dict"]):
        c.load_state_dict(sd)
    for c in classifiers:
        for p in c.parameters():
            p.requires_grad = False
    logger.info("Classifiers loaded and frozen.")
    return [None] * len(classifiers)


def train(args, model, device, dataset_name, datasets_dir):
    start_time = datetime.now()

    # ---- Losses & optimizer ----
    active_losses = args['losses']
    logger.info(f"Active losses: {active_losses}")

    if "ce" in active_losses:
        ce_criterion = torch.nn.CrossEntropyLoss()
    
    if "uncertainty" in active_losses and args['model_mode'] == "uncertainty":
        uncertainty_loss_type = args.get('uncertainty_loss', 'gaussian_nll').lower()
        if uncertainty_loss_type == 'gaussian_cosine':
            uncertainty_criterion = GaussianCosineLoss()
        else:  # default to gaussian_nll
            uncertainty_criterion = torch.nn.GaussianNLLLoss()
        uncertainty_lambda = args.get('uncertainty_lambda', 1.0)
        logger.info(f"Using uncertainty loss: {uncertainty_loss_type}")
        if args.get('separate_variance_aggregation'):
            logger.info("Using separate aggregation for variance.")
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

    train_set_folder = f"{datasets_dir[dataset_name]['train']}"
    val_set_folder = f"{datasets_dir[dataset_name]['validation']}"

    # ---- Datasets & classifiers ----
    groups = [TrainDataset(dataset_name, args, train_set_folder, M=args['M'], alpha=args['alpha'], N=args['N'], L=args['L'],
                        current_group=n, min_images_per_class=args['min_images_per_class']) for n in range(args['groups_num'])]
    # Each group has its own classifier, which depends on the number of classes in the group
    classifiers = [cosface_loss.MarginCosineProduct(args['descriptors_dimension'], len(group)) for group in groups]
    classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args['classifiers_lr']) for classifier in classifiers]

    logger.info(f"Using {len(groups)} groups")
    logger.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
    logger.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

    val_ds = TestDataset(f"{val_set_folder}/database", f"{val_set_folder}/queries", args['positive_dist_threshold'], args.get('image_size'), use_labels=True)
    logger.info(f"Validation set: {val_ds}")

    # GPU augmentations operate on batches (4D tensors) and must be applied in the
    # training loop after collation, unlike CPU augmentations which run per-image
    # inside TrainDataset.__getitem__.
    if args['augmentation_device'] == "cuda":
        gpu_augmentation = T.Compose([
                augmentations.DeviceAgnosticColorJitter(brightness=args['brightness'],
                                                        contrast=args['contrast'],
                                                        saturation=args['saturation'],
                                                        hue=args['hue']),
                augmentations.DeviceAgnosticRandomResizedCrop([args['image_size'], args['image_size']],
                                                            scale=[1-args['random_resized_crop'], 1]),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    # ---- Resume from checkpoint ----
    if args.get('resume_train') is not None:
        model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
            util.resume_train(device, args, args['log_dir'], model, model_optimizer, classifiers, classifiers_optimizers)
        epoch_num = start_epoch_num - 1
        logger.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args['resume_train']}")
    elif args.get('load_classifiers'):
        best_val_recall1 = start_epoch_num = 0
        resume_path = args.get('resume_model')
        if resume_path is None:
            logger.warning("--load_classifiers set but no --resume_model; skipping.")
        else:
            logger.info(f"Loading classifier weights from {resume_path}")
            frozen_optims = _load_and_freeze_classifiers(classifiers, resume_path)
            if frozen_optims is not None:
                classifiers_optimizers = frozen_optims
    else:
        best_val_recall1 = start_epoch_num = 0

    if args.get('resume_train') or args.get('resume_model'):
        logger.info("Verifying resumed model performance...")
        _, resume_recalls_str, _ = eval_dataset(args, model, device, dataset_name, val_set_folder)
        logger.info(f"Resumed model performance: {resume_recalls_str}")

    # ---- Training loop ----
    logger.info("Start training ...")
    logger.info(f"There are {len(groups[0])} classes for the first group, " +
                f"each epoch has {args['iterations_per_epoch']} iterations " +
                f"with batch_size {args['batch_size']}, therefore the model sees each class (on average) " +
                f"{args['iterations_per_epoch'] * args['batch_size'] / len(groups[0]):.1f} times per epoch")

    patience = args.get('patience', 5)
    not_improved_count = 0

    mean_variances_history = []
    for epoch_num in range(start_epoch_num, args['epochs_num']):
        
        # ---- Train ----
        epoch_start_time = datetime.now()
        current_group_num = epoch_num % args['groups_num']
        classifiers[current_group_num] = classifiers[current_group_num].to(device)
        if classifiers_optimizers[current_group_num] is not None:
            util.move_to_device(classifiers_optimizers[current_group_num], device)

        dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args['num_workers'],
                                                batch_size=args['batch_size'], shuffle=True,
                                                pin_memory=(device == "cuda"), drop_last=True)

        dataloader_iterator = iter(dataloader)
        model = model.train()
        
        epoch_losses = []
        epoch_losses_ce = []
        epoch_losses_gnll = []
        epoch_variances = []
        
        for iteration in tqdm(range(args['iterations_per_epoch']), ncols=100):
            images, targets, _ = next(dataloader_iterator)
            images, targets = images.to(device), targets.to(device)
            
            if args['augmentation_device'] == "cuda":
                images = gpu_augmentation(images)
            
            model_optimizer.zero_grad()
            if classifiers_optimizers[current_group_num] is not None:
                classifiers_optimizers[current_group_num].zero_grad()
            
            loss = torch.tensor(0.0, device=device)

            mu_norm, variance = model(images)
            if "ce" in active_losses:
                output = classifiers[current_group_num](mu_norm, targets)
                loss_ce = ce_criterion(output, targets)
                loss = loss + loss_ce
                epoch_losses_ce.append(loss_ce.item())

            # Uncertainty calculations (outside autocast for stability)
            if "uncertainty" in active_losses and args['model_mode'] == "uncertainty":
                # Normalize class weights to get the ground-truth "prototype" for each class
                weights = classifiers[current_group_num].weight
                norm_weights = torch.nn.functional.normalize(weights, p=2, dim=1)
                # Select the specific target vector for each image in the batch
                target_vectors = norm_weights[targets]
                    
                # Calculate uncertainty loss comparing the descriptor to its class prototype
                loss_uncertainty = uncertainty_criterion(mu_norm, target_vectors, variance)
                
                # Sum of classification loss and uncertainty estimation loss
                loss = loss + uncertainty_lambda * loss_uncertainty
                epoch_losses_gnll.append((uncertainty_lambda * loss_uncertainty).item())
            
            if args['model_mode'] == "uncertainty":
                epoch_variances.append(variance.mean().item())
                
            # ---- Backward ----
            loss.backward()
            model_optimizer.step()
            if classifiers_optimizers[current_group_num] is not None:
                classifiers_optimizers[current_group_num].step()
            
            epoch_losses.append(loss.item())
            del loss
            if "ce" in active_losses:
                del output
            del images
            
            if args['dry_run']:
                logger.info("Dry run: breaking epoch loop after one iteration")
                break
        
        classifiers[current_group_num] = classifiers[current_group_num].cpu()
        if classifiers_optimizers[current_group_num] is not None:
            util.move_to_device(classifiers_optimizers[current_group_num], "cpu")
        
        if args['model_mode'] == "uncertainty":
            mean_total_loss = np.mean(epoch_losses)
            mean_variance = np.mean(epoch_variances)
            mean_variances_history.append(mean_variance)
            
            log_msg = f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, loss_total = {mean_total_loss:.4f}"
            if "ce" in active_losses:
                log_msg += f", loss_ce = {np.mean(epoch_losses_ce):.4f}"
            if "uncertainty" in active_losses:
                log_msg += f", loss_uncertainty = {np.mean(epoch_losses_gnll):.4f}"
                log_msg += f", mean_variance = {mean_variance:.4f}"
            logger.info(log_msg)
        else:
            logger.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                        f"loss = {np.mean(epoch_losses):.4f}")
        
        # ---- Evaluate ----
        recalls, recalls_str, _ = eval_dataset(args, model, device, dataset_name, val_set_folder)
        logger.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {recalls_str}")
        is_best = recalls[0] > best_val_recall1
        best_val_recall1 = max(recalls[0], best_val_recall1)
        
        if is_best:
            not_improved_count = 0
        else:
            not_improved_count += 1
            if not_improved_count >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

        # Save checkpoint, which contains all training parameters
        if not args['dry_run']:
            util.save_checkpoint({
                "epoch_num": epoch_num + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": model_optimizer.state_dict(),
                "classifiers_state_dict": [c.state_dict() for c in classifiers],
                "optimizers_state_dict": [c.state_dict() if c is not None else {} for c in classifiers_optimizers],
                "best_val_recall1": best_val_recall1
            }, is_best, args['log_dir'])

        if args['dry_run']:
            break

    if mean_variances_history:
        logger.info(f"Mean variance evolution: start={mean_variances_history[0]:.4f}, end={mean_variances_history[-1]:.4f}")

    logger.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")
    logger.info("Experiment finished (without any errors)")

if __name__ == "__main__":
    # ---- Load config and datasets (shared helper) ----
    cfg, entries, datasets_dir = build_config_and_datasets()

    # Training-specific setup
    commons.make_deterministic(cfg["seed"])
    # Handle the cuDNN Benchmark speed/reproducibility trade-off
    commons.setup_cudnn(cfg["cudnn_benchmark"])
    device, model = init_model(cfg)

    logger.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")
    if torch.cuda.is_available():
        logger.info(f"GPU type: {torch.cuda.get_device_name(0)}")

    # Optionally copy the resume model into the current log directory
    commons.copy_resume_model_to_log_dir(cfg, logger)

    for e in entries:
        logger.info(f"Training dataset: {e['name']}")
        train(cfg, model, device, e["name"], datasets_dir)
    logger.info("Training completed.")