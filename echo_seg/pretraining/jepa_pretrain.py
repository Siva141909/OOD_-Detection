# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Echo Frame-level JEPA Pretraining Script
# Pretrains on EchoNet-Dynamic ED/ES frames

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

# Add echo_seg to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.echo_dataset import make_echo_pretrain_dataset


# Setup logging with file handler - FORCE FLUSH
class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Remove existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            FlushFileHandler(log_file)  # Force flush after each log
        ]
    )
    return log_file

logger = logging.getLogger(__name__)


def get_gpu_memory_info():
    """Get current GPU memory usage and temperature if available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        # Try to get GPU temperature (requires pynvml)
        temp_str = ""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            temp_str = f", Temp: {temp}°C"
            pynvml.nvmlShutdown()
        except:
            pass
        
        return f"GPU Mem: {allocated:.2f}GB alloc, {reserved:.2f}GB reserved, {max_allocated:.2f}GB max{temp_str}"
    return "CUDA not available"


def clear_gpu_memory():
    """Clear GPU cache and sync."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset max memory stats for cleaner monitoring
        torch.cuda.reset_peak_memory_stats()


def parse_args():
    parser = argparse.ArgumentParser(description='Echo JEPA Single-GPU Training')
    parser.add_argument(
        '--config', type=str,
        default='configs/echo_vit_small.yaml',
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
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory and setup file logging
    output_dir = config['logging']['folder']
    os.makedirs(output_dir, exist_ok=True)
    log_file = setup_logging(output_dir)
    
    logger.info(f'Log file: {log_file}')
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
    if torch.cuda.is_available():
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    
    # Save config to output dir
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Gradient accumulation steps
    grad_accum_steps = config['data'].get('gradient_accumulation_steps', 1)
    logger.info(f'Gradient accumulation steps: {grad_accum_steps}')
    
    # ==================== DATA ====================
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
    dataset, dataloader, _ = make_echo_pretrain_dataset(
        transform=transform,
        batch_size=batch_size,
        collator=mask_collator,
        pin_mem=config['data']['pin_mem'],
        num_workers=config['data']['num_workers'],
        root_path=config['data']['root_path'],
        split="TRAIN",
        frame_type="both",
        img_size=crop_size,
        drop_last=True,
        shuffle=True
    )
    
    logger.info(f'Dataset size: {len(dataset)} images')
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
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=config['optimization']['weight_decay'],
        final_wd=config['optimization']['final_weight_decay'],
        start_lr=config['optimization']['start_lr'],
        ref_lr=config['optimization']['lr'],
        final_lr=config['optimization']['final_lr'],
        iterations_per_epoch=len(dataloader),
        warmup=config['optimization']['warmup'],
        num_epochs=config['optimization']['epochs'],
        ipe_scale=config['optimization']['ipe_scale'],
        use_bfloat16=config['meta']['use_bfloat16']
    )
    
    # EMA momentum scheduler
    ema = config['optimization']['ema']
    ema_start, ema_end = ema[0], ema[1]
    
    # ==================== TRAINING ====================
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            args.resume, encoder, predictor, target_encoder, 
            optimizer, scaler, device
        )
    
    # Training tracking
    all_losses = []
    all_embedding_stats = []
    
    num_epochs = config['optimization']['epochs']
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f'\n{"="*50}')
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        logger.info(f'{get_gpu_memory_info()}')
        logger.info(f'{"="*50}')
        
        encoder.train()
        predictor.train()
        target_encoder.train()
        
        loss_meter = AverageMeter()
        optimizer.zero_grad()  # Zero grads at start
        
        try:
            for itr, (imgs, masks_enc, masks_pred) in enumerate(dataloader):
                # Move data to device
                imgs = imgs.to(device, non_blocking=True)
                masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
                masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]
                
                B = imgs.shape[0]  # Batch size
                
                # Calculate EMA momentum for this step
                total_steps = num_epochs * len(dataloader)
                current_step = epoch * len(dataloader) + itr
                momentum = ema_end - (ema_end - ema_start) * (1 - current_step / total_steps)
                
                # Forward pass with gradient accumulation
                try:
                    # Sync before forward pass
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    # Get target representations (no gradient)
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                    
                    # Get context representations - DISABLE mixed precision for stability
                    z = encoder(imgs, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    loss = F.mse_loss(z, h) / grad_accum_steps
                    loss.backward()

                    # Sync after backward
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    # Step optimizer every grad_accum_steps
                    if (itr + 1) % grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        # Update learning rate and weight decay
                        scheduler.step()
                        wd_scheduler.step()
                        
                        # EMA update of target encoder
                        with torch.no_grad():
                            for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                                param_k.data.mul_(momentum).add_((1. - momentum) * param_q.detach().data)
                    
                    # Log (scale loss back to actual value)
                    actual_loss = loss.item() * grad_accum_steps
                    loss_meter.update(actual_loss)
                    all_losses.append(actual_loss)
                    
                    # Clear intermediate tensors
                    del z, h
                    
                    if itr % 25 == 0:  # Log more frequently
                        lr = optimizer.param_groups[0]['lr']
                        logger.info(f'  Iter {itr}/{len(dataloader)}: loss={actual_loss:.4f}, lr={lr:.6f}, mom={momentum:.4f}')
                        logger.info(f'    {get_gpu_memory_info()}')
                        # Clear cache every log (25 batches)
                        clear_gpu_memory()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"CUDA OOM at iter {itr}! {get_gpu_memory_info()}")
                        clear_gpu_memory()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise e
        
        except Exception as e:
            logger.error(f"Error during epoch {epoch+1}: {str(e)}")
            logger.error(f"{get_gpu_memory_info()}")
            # Save emergency checkpoint
            save_checkpoint(
                encoder, predictor, target_encoder, optimizer, scaler,
                epoch, loss_meter.avg if loss_meter.count > 0 else 0,
                os.path.join(output_dir, f'emergency_checkpoint_ep{epoch}.pth')
            )
            raise e
        
        # End of epoch
        logger.info(f'Epoch {epoch+1} complete. Avg loss: {loss_meter.avg:.4f}')
        
        # Compute embedding stats
        stats = compute_embedding_stats(encoder, target_encoder, dataloader, device)
        stats['epoch'] = epoch + 1
        all_embedding_stats.append(stats)
        logger.info(f'Embedding stats: var={stats["embedding_var"]:.4f}, std={stats["embedding_std"]:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            save_checkpoint(
                encoder, predictor, target_encoder, optimizer, scaler,
                epoch + 1, loss_meter.avg,
                os.path.join(output_dir, f'checkpoint_ep{epoch+1}.pth')
            )
            
            # Plot training curves
            plot_training_curves(all_losses, all_embedding_stats, output_dir)
    
    logger.info('\nTraining complete!')
    logger.info(f'Final model saved to {output_dir}')


if __name__ == '__main__':
    main()
