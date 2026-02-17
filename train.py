
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T
import shutil
from pathlib import Path

from configs.parser import build_config, init_model
from data.test_dataset import TestDataset
from data.train_dataset import TrainDataset
from data.upload_dataset import upload_dataset
from eval import eval_dataset
from losses import cosface_loss
from losses.cosface_loss import cosine_distance
from losses.gaussian_cosine_loss import GaussianCosineLoss
from utils import augmentations, commons, util


# Define the logger for this module
# It will inherit the configuration set in setup_logging within parser.py
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True  # Provides a speedup

def train(args, model, device, dataset_name, datasetsts_dir):
    start_time = datetime.now()

    logger.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")
    if torch.cuda.is_available():
        logger.info(f"GPU type: {torch.cuda.get_device_name(0)}")

    #### Optimizer
    active_losses = args['losses']
    if active_losses is None:
        if args['model_mode'] == "uncertainty":
            active_losses = ["ce", "uncertainty"]
        else:
            active_losses = ["ce"]
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

    train_set_folder = f"{datasetsts_dir[dataset_name]['train']}"
    val_set_folder = f"{datasetsts_dir[dataset_name]['validation']}"

    #### Datasets
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

    #### Resume
    if args.get('resume_train') is not None:
        model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
            util.resume_train(device, args, args['log_dir'], model, model_optimizer, classifiers, classifiers_optimizers)
        
        if not args.get('load_classifiers'):
            epoch_num = start_epoch_num - 1
            logger.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args['resume_train']}")
            
            # Verify resume performance
            logger.info("Verifying resumed model performance...")
            _, resume_recalls_str, _ = eval_dataset(args, model, device, dataset_name, val_set_folder)
            logger.info(f"Resumed model performance: {resume_recalls_str}")
    elif args.get('load_classifiers'):
        best_val_recall1 = start_epoch_num = 0
        resume_path = args['resume_model'] if args.get('resume_model') is not None else args['resume_train']
        logger.info(f"Loading ONLY classifier weights from {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu')
        if isinstance(checkpoint, dict) and "classifiers_state_dict" in checkpoint:
            if len(checkpoint["classifiers_state_dict"]) == len(classifiers):
                for c, sd in zip(classifiers, checkpoint["classifiers_state_dict"]):
                    c.load_state_dict(sd)
                logger.info("Classifiers weights loaded successfully.")
            else:
                logger.warning(f"Skipping classifiers load: Checkpoint has {len(checkpoint['classifiers_state_dict'])} classifiers, config has {len(classifiers)}.")
        else:
            logger.warning("No classifiers_state_dict found in checkpoint.")
    elif args.get('resume_model') is not None:
        best_val_recall1 = start_epoch_num = 0
        logger.info(f"Resuming from model {args['resume_model']}")
        logger.info("Verifying resumed model performance...")
        _, resume_recalls_str, _ = eval_dataset(args, model, device, dataset_name, val_set_folder)
        logger.info(f"Resumed model performance: {resume_recalls_str}")
    else:
        best_val_recall1 = start_epoch_num = 0

    #### Train / evaluation loop
    logger.info("Start training ...")
    logger.info(f"There are {len(groups[0])} classes for the first group, " +
                f"each epoch has {args['iterations_per_epoch']} iterations " +
                f"with batch_size {args['batch_size']}, therefore the model sees each class (on average) " +
                f"{args['iterations_per_epoch'] * args['batch_size'] / len(groups[0]):.1f} times per epoch")


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

    if args['use_amp16']:
        scaler = torch.amp.GradScaler("cuda")

    patience = args.get('patience', 5)
    not_improved_count = 0

    mean_variances_history = []
    for epoch_num in range(start_epoch_num, args['epochs_num']):
        
        #### Train
        epoch_start_time = datetime.now()
        # Select classifier and dataloader according to epoch
        current_group_num = epoch_num % args['groups_num']
        classifiers[current_group_num] = classifiers[current_group_num].to(device)
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
            
            if args['augmentation_device']  == "cuda":
                images = gpu_augmentation(images)
            
            model_optimizer.zero_grad()
            classifiers_optimizers[current_group_num].zero_grad()
            
            loss = torch.tensor(0.0, device=device)

            # Forward pass and classifier loss (with AMP if enabled)
            if args['use_amp16']:
                with torch.amp.autocast("cuda"):
                    mu_norm, variance = model(images)
                    if "ce" in active_losses:
                        output = classifiers[current_group_num](mu_norm, targets)
                        loss_ce = ce_criterion(output, targets)
                        loss = loss + loss_ce
            else:
                mu_norm, variance = model(images)
                if "ce" in active_losses:
                    output = classifiers[current_group_num](mu_norm, targets)
                    loss_ce = ce_criterion(output, targets)
                    loss = loss + loss_ce
            
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

            if "ce" in active_losses:
                epoch_losses_ce.append(loss_ce.item())
            
            # Backward pass
            if args['use_amp16']:
                scaler.scale(loss).backward()
                scaler.step(model_optimizer)
                scaler.step(classifiers_optimizers[current_group_num])
                scaler.update()
            else:
                loss.backward()
                model_optimizer.step()
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
        
        #### Evaluation
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
                "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
                "best_val_recall1": best_val_recall1
            }, is_best, args['log_dir'])

        if args['dry_run']:
            break

    if mean_variances_history:
        logger.info(f"Mean variance evolution: start={mean_variances_history[0]:.4f}, end={mean_variances_history[-1]:.4f}")

    logger.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")
    logger.info("Experiment finished (without any errors)")

if __name__ == "__main__":
    # ---- Load and build config ----       
    cfg, entries = build_config()

    datasetsts_dir = upload_dataset(cfg, entries)
    commons.make_deterministic(cfg['seed'])
    # Handle the cuDNN Benchmark speed/reproducibility trade-off
    commons.setup_cudnn(cfg['cudnn_benchmark'])
    device, model = init_model(cfg)

    if cfg.get('resume_model'):
        src = Path(cfg['resume_model'])
        if src.exists():
            shutil.copy(src, cfg['log_dir'])
            logger.info(f"Copied resume model from {src} to {cfg['log_dir']}")

    for e in entries:
        logger.info(f"Training dataset: {e['name']}")
        train(cfg, model, device, e["name"], datasetsts_dir)
    logger.info("Training completed.")