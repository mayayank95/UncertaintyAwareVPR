
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T

from configs.parser import build_config
from data.test_dataset import TestDataset
from data.train_dataset import TrainDataset
from data.upload_dataset import upload_dataset
from eval import eval_dataset
from models.get_model import get_model
from losses import cosface_loss
from utils import augmentations, commons, util


# Define the logger for this module
# It will inherit the configuration set in setup_logging within parser.py
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True  # Provides a speedup

def init(args):
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(
        f"Training with {args['method']} with a {args['backbone']} backbone and descriptors dimension {args['descriptors_dimension']}"
    )
    logger.info(f"The outputs are being saved in {args['log_dir']}")

    #model = get_model(args['method'], args['backbone'], args['descriptors_dimension'], args.get('resume_model'), args.get('train_all_layers', False))
    model = get_model(args)
    device = torch.device(args["device"])
    model = model.to(device) 
    model.train()
    return device, model

def train(args, model, device, dataset_name, datasetsts_dir):
    start_time = datetime.now()
    commons.make_deterministic(args['seed'])
    # Handle the cuDNN Benchmark speed/reproducibility trade-off
    if args['cudnn_benchmark']:
        # If speed is requested:
        torch.backends.cudnn.benchmark = True 
        torch.backends.cudnn.deterministic = False
        logger.info("cuDNN benchmark ENABLED: Training will be FASTER but not bit-by-bit reproducible.")
    else:
        # If reproducibility is requested (already set to False by make_deterministic):
        # This ensures exact results if the same seed is used
        logger.info("cuDNN benchmark DISABLED: Training will be bit-by-bit DETERMINISTIC (Slower).")

    logger.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

    #### Optimizer
    ce_criterion = torch.nn.CrossEntropyLoss()
    if args['model_mode'] == "uncertainty":
        gnll_criterion = torch.nn.GaussianNLLLoss()
        uncertainty_lambda = args.get('uncertainty_lambda', 1.0)
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
        epoch_num = start_epoch_num - 1
        logger.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args['resume_train']}")
        
        # Verify resume performance
        logger.info("Verifying resumed model performance...")
        _, resume_recalls_str = eval_dataset(args, model, device, dataset_name, val_set_folder)
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
        
        epoch_losses = []#np.zeros((0, 1), dtype=np.float32)
        for iteration in tqdm(range(args['iterations_per_epoch']), ncols=100):
            images, targets, _ = next(dataloader_iterator)
            images, targets = images.to(device), targets.to(device)
            
            if args['augmentation_device']  == "cuda":
                images = gpu_augmentation(images)
            
            model_optimizer.zero_grad()
            classifiers_optimizers[current_group_num].zero_grad()
            
            if not args['use_amp16']:
                descriptors, vars = model(images)
                output = classifiers[current_group_num](descriptors, targets)
                #loss = criterion1(output, targets)
                loss_ce = ce_criterion(output, targets)
                if args['model_mode'] == "uncertainty":
                    # Normalize class weights to get the ground-truth "prototype" for each class
                    weights = classifiers[current_group_num].weight
                    norm_weights = torch.nn.functional.normalize(weights, p=2, dim=1)
                    # Select the specific target vector for each image in the batch
                    target_vectors = norm_weights[targets]                    
                    # Calculate GNLL loss comparing the descriptor to its class prototype
                    # vars is log_sigma_sq, so we exponentiate it to get variance
                    loss_gnll = gnll_criterion(descriptors, target_vectors, torch.exp(vars))
                    # Sum of classification loss and uncertainty estimation loss
                    loss = loss_ce + uncertainty_lambda * loss_gnll
                else:
                    loss = loss_ce
                loss.backward()
                # epoch_losses = np.append(epoch_losses, loss.item())
                epoch_losses.append(loss.item())
                del loss, output, images
                model_optimizer.step()
                classifiers_optimizers[current_group_num].step()
            else:  # Use AMP 16
                with torch.amp.autocast("cuda"):
                    descriptors, vars = model(images)
                    output = classifiers[current_group_num](descriptors, targets)
                    # loss = criterion1(output, targets)
                    loss_ce = ce_criterion(output, targets)
                    if args['model_mode'] == "uncertainty":
                        # Normalize class weights to get the ground-truth "prototype" for each class
                        weights = classifiers[current_group_num].weight
                        norm_weights = torch.nn.functional.normalize(weights, p=2, dim=1)
                        # Select the specific target vector for each image in the batch
                        target_vectors = norm_weights[targets]                    
                        # Calculate GNLL loss comparing the descriptor to its class prototype
                        # vars is log_sigma_sq, so we exponentiate it to get variance
                        loss_gnll = gnll_criterion(descriptors, target_vectors, torch.exp(vars))
                        # Sum of classification loss and uncertainty estimation loss
                        loss = loss_ce + uncertainty_lambda * loss_gnll
                    else:
                        loss = loss_ce
                scaler.scale(loss).backward()
                # epoch_losses = np.append(epoch_losses, loss.item())
                epoch_losses.append(loss.item())
                del loss, output, images
                scaler.step(model_optimizer)
                scaler.step(classifiers_optimizers[current_group_num])
                scaler.update()
            
            if args['dry_run']:
                logger.info("Dry run: breaking epoch loop after one iteration")
                break
        
        classifiers[current_group_num] = classifiers[current_group_num].cpu()
        util.move_to_device(classifiers_optimizers[current_group_num], "cpu")
        
        # logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
        #             f"loss = {epoch_losses.mean():.4f}")
        logger.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                    f"loss = {np.mean(epoch_losses):.4f}")
        
        #### Evaluation
        recalls, recalls_str = eval_dataset(args, model, device, dataset_name, val_set_folder)
        logger.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
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

    logger.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")
    logger.info("Experiment finished (without any errors)")

if __name__ == "__main__":
    # ---- Load and build config ----       
    cfg, entries = build_config()
    datasetsts_dir = upload_dataset(cfg, entries)
    device, model = init(cfg)
    for e in entries:
        logger.info(f"Training dataset: {e['name']}")
        train(cfg, model, device, e["name"], datasetsts_dir)
    logger.info("Training completed.")