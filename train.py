import logging
import multiprocessing
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm

from configs.runtime import build_config_and_datasets, init_model, init_wandb
from data.test_dataset import TestDataset
from data.train_dataset import TrainDataset
from eval import eval_dataset
from losses import cosface_loss
from losses import uncertainty_utils
from utils import augmentations, commons, util, wandb_utils

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
        uncertainty_lambda = args.get('uncertainty_lambda', 1.0)
        logger.info(f"Using uncertainty loss: {args.get('uncertainty_loss', 'gaussian_nll')}")
        logger.info(f"Variance head type: {args.get('var_head_type', 'linear')}")

    # LR for model: when freeze_model use head_lr (default 1e-3) so uncertainty head can learn; else use --lr
    model_lr = (args.get("head_lr") if args.get("head_lr") is not None else 1e-3) if args.get("freeze_model") else args["lr"]
    model_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=model_lr)
    opt_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    logger.debug(f"Optimizer (model): lr={model_lr}, params ({len(opt_param_names)}): {opt_param_names}")
    if args.get("freeze_model"):
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"freeze_model: training {n_trainable} params (backbone/aggregation frozen)")

    train_set_folder = f"{datasets_dir[dataset_name]['train']}"
    val_set_folder = f"{datasets_dir[dataset_name]['validation']}"

    # ---- Datasets & classifiers ----
    groups = [TrainDataset(dataset_name, args, train_set_folder, M=args['M'], alpha=args['alpha'], N=args['N'], L=args['L'],
                        current_group=n, min_images_per_class=args['min_images_per_class']) for n in range(args['groups_num'])]
    # Each group has its own classifier, which depends on the number of classes in the group
    classifiers = [cosface_loss.MarginCosineProduct(args['descriptors_dimension'], len(group)) for group in groups]
    classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args['classifiers_lr']) for classifier in classifiers]

    logger.debug(f"Using {len(groups)} groups")
    logger.debug(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
    logger.debug(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

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
        logger.info(f"Loading classifier weights from {args['load_classifiers']}")
        frozen_optims = _load_and_freeze_classifiers(classifiers, args['load_classifiers'])
        if frozen_optims is not None:
            classifiers_optimizers = frozen_optims
    else:
        best_val_recall1 = start_epoch_num = 0

    early_stop_metric = args.get("early_stop_metric", "recall")
    best_val_gnll = float("inf")
    if args.get('resume_train') or args.get('resume_model'):
        logger.info("Verifying resumed model performance (before any training)...")
        init_recalls, _, init_map_at_k, init_corr, init_mean_var, init_std_var, init_min_var, init_max_var, _, _, init_val_gnll = eval_dataset(
            args, model, device, dataset_name, val_set_folder, log_dataset_info=False,
        )
        _mv = f"{init_mean_var:.4f}" if init_mean_var is not None else "N/A"
        _xv = f"{init_max_var:.4f}" if init_max_var is not None else "N/A"
        _gnll = f", val_gnll={init_val_gnll:.4f}" if init_val_gnll is not None else ""
        _map = f", mAP@1={init_map_at_k[0]:.2f}" if init_map_at_k is not None else ""
        logger.info(f"Initial val (after var_init, before training): R@1={init_recalls[0]:.1f}{_map}, mean_var={_mv}, max_var={_xv}{_gnll}")

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
                target_vectors = norm_weights[targets]
                loss_uncertainty = uncertainty_utils.compute_uncertainty_loss(
                    mu_norm, target_vectors, variance,
                    loss_type=args.get('uncertainty_loss', 'gaussian_nll'),
                    lambda_=uncertainty_lambda,
                )
                loss = loss + loss_uncertainty
                epoch_losses_gnll.append(loss_uncertainty.item())
            
            if args['model_mode'] == "uncertainty":
                epoch_variances.append(variance.mean().item())
                
            # ---- Backward ----
            loss.backward()
            if args.get("debug_var_head_grad") and args.get("model_mode") == "uncertainty" and iteration == 0:
                root = getattr(model, "module", model)
                if hasattr(root, "var_head"):
                    norms = []
                    for n, p in root.var_head.named_parameters():
                        g = p.grad.norm().item() if p.grad is not None else 0.0
                        norms.append(f"{n}={g:.6f}")
                    logger.debug(f"[debug_var_head_grad] epoch {epoch_num} first batch var_head grad norms: {norms}")
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
        recalls, _, map_at_k, uncertainty_corr, mean_query_variance, std_query_variance, min_query_variance, max_query_variance, eval_wandb_metrics, eval_wandb_images, val_gnll = eval_dataset(
            args, model, device, dataset_name, val_set_folder, wandb_step=epoch_num, log_dataset_info=False,
        )
        if early_stop_metric == "val_gnll" and val_gnll is not None:
            is_best = val_gnll < best_val_gnll
            best_val_gnll = min(val_gnll, best_val_gnll)
        else:
            is_best = recalls[0] > best_val_recall1
            best_val_recall1 = max(recalls[0], best_val_recall1)

        wandb_utils.log_train_epoch(
            args, epoch_num, recalls, map_at_k, best_val_recall1, active_losses,
            epoch_variances, epoch_losses, epoch_losses_ce, epoch_losses_gnll,
            uncertainty_corr, mean_query_variance, std_query_variance, min_query_variance, max_query_variance,
            eval_wandb_metrics, eval_wandb_images,
        )

        if is_best:
            not_improved_count = 0
        else:
            not_improved_count += 1
            if not_improved_count >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement (metric: {early_stop_metric}).")
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
    init_wandb(cfg, job_type="train")

    # Training-specific setup
    commons.make_deterministic(cfg["seed"])
    # Handle the cuDNN Benchmark speed/reproducibility trade-off
    commons.setup_cudnn(cfg["cudnn_benchmark"])
    device, model = init_model(cfg)

    logger.debug(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")
    if torch.cuda.is_available():
        logger.debug(f"GPU type: {torch.cuda.get_device_name(0)}")

    # Optionally copy the resume model into the current log directory
    commons.copy_resume_model_to_log_dir(cfg, logger)

    for e in entries:
        logger.info(f"Training dataset: {e['name']}")
        train(cfg, model, device, e["name"], datasets_dir)
    logger.info("Training completed.")

    wandb_utils.finish_train_run(cfg)