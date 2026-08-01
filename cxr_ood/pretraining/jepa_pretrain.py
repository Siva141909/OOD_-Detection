# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Simplified single-GPU training script for Medical JEPA
# Easier to debug and monitor than distributed training

import os
import sys
import copy
import logging
import argparse
import pprint
from datetime import datetime

import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# Add IJEPA_Meta to path for core modules
IJEPA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "IJEPA_Meta")
sys.path.insert(0, IJEPA_PATH)

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.logging import AverageMeter
from src.utils.tensors import repeat_interleave_batch
from src.helper import init_model, init_opt
from src.transforms import make_transforms

# Import CXR dataset from our utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cxr_dataset import make_cxr_dataset


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Medical JEPA Single-GPU Training')
    parser.add_argument(
        '--config', type=str,
        default='configs/cxr_vit_small.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--device', type=str,
        default='cuda:0',
        help='Device to use (cuda:0, cuda:1, cpu)'
    )
    parser.add_argument(
        '--resume', type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--max_images', type=int,
        default=None,
        help='Max images to use (for debugging)'
    )
    return parser.parse_args()


def compute_embedding_stats(encoder, target_encoder, dataloader, device, num_batches=10):
    """
    Compute embedding statistics to monitor for collapse.
    
    Returns:
        dict with mean, std, and variance of embeddings
    """
    encoder.eval()
    target_encoder.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for i, (imgs, _, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # Handle both tensor and tuple formats from collator
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0].to(device)
            else:
                imgs = imgs.to(device)
            
            # Get target encoder embeddings (without masking)
            h = target_encoder(imgs)
            h = F.layer_norm(h, (h.size(-1),))
            
            # Take mean over patches to get image-level embedding
            h_mean = h.mean(dim=1)  # [B, D]
            all_embeddings.append(h_mean.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]
    
    # Compute statistics
    mean = all_embeddings.mean(dim=0)
    std = all_embeddings.std(dim=0)
    
    stats = {
        'embedding_mean': mean.mean().item(),
        'embedding_std': std.mean().item(),
        'embedding_var': (std ** 2).mean().item(),
        'feature_std_min': std.min().item(),
        'feature_std_max': std.max().item(),
    }
    
    encoder.train()
    target_encoder.train()
    
    return stats


def save_checkpoint(encoder, predictor, target_encoder, optimizer, scaler, epoch, loss, save_path):
    """Save training checkpoint."""
    checkpoint = {
        'encoder': encoder.state_dict(),
        'predictor': predictor.state_dict(),
        'target_encoder': target_encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    logger.info(f'Saved checkpoint to {save_path}')


def load_checkpoint(path, encoder, predictor, target_encoder, optimizer, scaler, device):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    encoder.load_state_dict(checkpoint['encoder'])
    predictor.load_state_dict(checkpoint['predictor'])
    target_encoder.load_state_dict(checkpoint['target_encoder'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scaler is not None and checkpoint['scaler'] is not None:
        scaler.load_state_dict(checkpoint['scaler'])
    
    epoch = checkpoint['epoch']
    logger.info(f'Loaded checkpoint from epoch {epoch}')
    
    return epoch


def plot_training_curves(losses, embedding_stats, save_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curve
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True)
    
    # Embedding variance (collapse detection)
    if embedding_stats:
        epochs = [s['epoch'] for s in embedding_stats]
        variances = [s['embedding_var'] for s in embedding_stats]
        stds = [s['embedding_std'] for s in embedding_stats]
        
        axes[0, 1].plot(epochs, variances, 'b-', label='Variance')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Embedding Variance')
        axes[0, 1].set_title('Embedding Variance (should NOT collapse to 0)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(epochs, stds, 'g-', label='Std')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Embedding Std')
        axes[1, 0].set_title('Embedding Std (should remain positive)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Loss histogram (last 100 iterations)
    if len(losses) > 100:
        axes[1, 1].hist(losses[-100:], bins=20)
        axes[1, 1].set_xlabel('Loss')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Loss Distribution (last 100 iters)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info('Config:')
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(config)
    
    # Setup device
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        logger.warning('CUDA not available, using CPU')
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f'Using device: {device}')
    
    # Create output directory
    output_dir = config['logging']['folder']
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config to output dir
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # ==================== DATA ====================
    # Extract data config
    batch_size = config['data']['batch_size']
    crop_size = config['data']['crop_size']
    crop_scale = tuple(config['data']['crop_scale'])
    
    # Create transforms
    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=config['data']['use_gaussian_blur'],
        horizontal_flip=config['data']['use_horizontal_flip'],
        color_distortion=config['data']['use_color_distortion'],
        color_jitter=config['data']['color_jitter_strength']
    )
    
    # Create mask collator
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=config['mask']['patch_size'],
        pred_mask_scale=tuple(config['mask']['pred_mask_scale']),
        enc_mask_scale=tuple(config['mask']['enc_mask_scale']),
        aspect_ratio=tuple(config['mask']['aspect_ratio']),
        nenc=config['mask']['num_enc_masks'],
        npred=config['mask']['num_pred_masks'],
        allow_overlap=config['mask']['allow_overlap'],
        min_keep=config['mask']['min_keep']
    )
    
    # Create dataset and dataloader
    # Get dataset selection from config (defaults to all if not specified)
    datasets_to_use = config['data'].get('datasets_to_use', None)
    
    dataset, dataloader, _ = make_cxr_dataset(
        transform=transform,
        batch_size=batch_size,
        collator=mask_collator,
        pin_mem=config['data']['pin_mem'],
        num_workers=config['data']['num_workers'],
        root_path=config['data']['root_path'],
        datasets_to_use=datasets_to_use,
        max_images=args.max_images,
        drop_last=True,
        shuffle=True
    )
    
    logger.info(f'Dataset size: {len(dataset)} images')
    logger.info(f'Datasets used: {datasets_to_use if datasets_to_use else "all"}')
    logger.info(f'Batches per epoch: {len(dataloader)}')
    
    # ==================== MODEL ====================
    encoder, predictor = init_model(
        device=device,
        patch_size=config['mask']['patch_size'],
        crop_size=crop_size,
        pred_depth=config['meta']['pred_depth'],
        pred_emb_dim=config['meta']['pred_emb_dim'],
        model_name=config['meta']['model_name']
    )
    
    # Create target encoder (EMA of encoder)
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    
    logger.info(f'Model: {config["meta"]["model_name"]}')
    logger.info(f'Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}')
    logger.info(f'Predictor parameters: {sum(p.numel() for p in predictor.parameters()):,}')
    
    # ==================== OPTIMIZER ====================
    num_epochs = config['optimization']['epochs']
    ipe = len(dataloader)  # iterations per epoch
    
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=float(config['optimization']['weight_decay']),
        final_wd=float(config['optimization']['final_weight_decay']),
        start_lr=config['optimization']['start_lr'],
        ref_lr=config['optimization']['lr'],
        final_lr=config['optimization']['final_lr'],
        iterations_per_epoch=ipe,
        warmup=config['optimization']['warmup'],
        num_epochs=num_epochs,
        ipe_scale=config['optimization']['ipe_scale'],
        use_bfloat16=config['meta']['use_bfloat16']
    )
    
    # EMA momentum scheduler
    ema = config['optimization']['ema']
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * config['optimization']['ipe_scale'])
        for i in range(int(ipe * num_epochs * config['optimization']['ipe_scale']) + 1)
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            args.resume, encoder, predictor, target_encoder, optimizer, scaler, device
        )
        # Fast-forward schedulers
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
    
    # ==================== TRAINING ====================
    logger.info('Starting training...')
    
    all_losses = []
    embedding_stats_history = []
    
    use_amp = config['meta']['use_bfloat16'] and device.type == 'cuda'
    
    for epoch in range(start_epoch, num_epochs):
        encoder.train()
        predictor.train()
        target_encoder.train()
        
        loss_meter = AverageMeter()
        
        for itr, (udata, masks_enc, masks_pred) in enumerate(dataloader):
            # Load data - handle both tensor and tuple formats
            if isinstance(udata, (list, tuple)):
                imgs = udata[0].to(device, non_blocking=True)
            else:
                imgs = udata.to(device, non_blocking=True)
            masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
            masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]
            
            # Update learning rate and weight decay
            _new_lr = scheduler.step()
            _new_wd = wd_scheduler.step()
            
            # Forward pass
            with autocast(enabled=use_amp):
                # Target encoder forward (no grad)
                with torch.no_grad():
                    h = target_encoder(imgs)
                    h = F.layer_norm(h, (h.size(-1),))
                    B = len(h)
                    h = apply_masks(h, masks_pred)
                    h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                
                # Context encoder forward
                z = encoder(imgs, masks_enc)
                z = predictor(z, masks_enc, masks_pred)
                
                # Loss
                loss = F.smooth_l1_loss(z, h)
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f'Loss is NaN/Inf at epoch {epoch+1}, iter {itr}')
                return
            
            # Backward pass
            optimizer.zero_grad()
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0)
                optimizer.step()
            
            # EMA update of target encoder
            with torch.no_grad():
                m = next(momentum_scheduler)
                for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                    param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)
            
            # Logging
            loss_val = loss.item()
            loss_meter.update(loss_val)
            all_losses.append(loss_val)
            
            if itr % 10 == 0:
                logger.info(
                    f'[Epoch {epoch+1}/{num_epochs}] [Iter {itr}/{ipe}] '
                    f'Loss: {loss_meter.avg:.4f} | LR: {_new_lr:.2e} | WD: {_new_wd:.2e}'
                )
        
        # End of epoch
        logger.info(f'Epoch {epoch+1} completed. Avg Loss: {loss_meter.avg:.4f}')
        
        # Compute embedding statistics (collapse detection)
        emb_stats = compute_embedding_stats(encoder, target_encoder, dataloader, device)
        emb_stats['epoch'] = epoch + 1
        embedding_stats_history.append(emb_stats)
        
        logger.info(
            f'Embedding stats: var={emb_stats["embedding_var"]:.4f}, '
            f'std={emb_stats["embedding_std"]:.4f}'
        )
        
        # Check for collapse
        if emb_stats['embedding_var'] < 1e-6:
            logger.warning('WARNING: Embedding variance is very low - possible collapse!')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                encoder, predictor, target_encoder, optimizer, scaler,
                epoch + 1, loss_meter.avg,
                os.path.join(output_dir, f'checkpoint_ep{epoch+1}.pth')
            )
        
        # Always save latest
        save_checkpoint(
            encoder, predictor, target_encoder, optimizer, scaler,
            epoch + 1, loss_meter.avg,
            os.path.join(output_dir, 'checkpoint_latest.pth')
        )
        
        # Plot training curves
        plot_training_curves(all_losses, embedding_stats_history, output_dir)
    
    logger.info('Training completed!')
    logger.info(f'Final checkpoint saved to {output_dir}')


if __name__ == '__main__':
    main()
