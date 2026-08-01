#!/usr/bin/env python3
"""
MAE (Masked Autoencoder) Pretraining on EchoNet-Dynamic

Uses vit_small() from JEPA codebase as encoder for EXACT architecture match.

Architecture:
- Encoder: vit_small (embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0)
- Decoder: Lightweight 4-layer transformer

This ensures identical encoder weights can be compared with JEPA.
"""

import os
import sys
import argparse
import json
import logging
import random
import math
from datetime import datetime
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Add IJEPA_Meta path for JEPA modules
IJEPA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "IJEPA_Meta")
sys.path.insert(0, IJEPA_PATH)
from src.models.vision_transformer import vit_small

# Add echo_seg to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.echo_dataset import EchoPretrainDataset, load_dynamic_data
from utils.common import DYNAMIC_PATH


# Force flush file handler
class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

def setup_logging(log_dir):
    """Setup logging with file handler - FORCE FLUSH."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Remove existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s:%(name)s:%(message)s',
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
        
        # Try to get GPU temperature
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
        torch.cuda.reset_peak_memory_stats()


# ============================================================================
# Position Embeddings (for decoder)
# ============================================================================

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 2D sincos position embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


# ============================================================================
# Decoder Components
# ============================================================================

class DecoderBlock(nn.Module):
    """Transformer block for decoder."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MAEDecoder(nn.Module):
    """Lightweight decoder for MAE reconstruction."""
    def __init__(self, num_patches, patch_size=16, in_chans=3,
                 encoder_embed_dim=384, decoder_embed_dim=192, 
                 decoder_depth=4, decoder_num_heads=6):
        super().__init__()
        
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.decoder_embed_dim = decoder_embed_dim
        
        # Project encoder dim to decoder dim
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Position embeddings for decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio=4., qkv_bias=True)
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Predict pixels
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans)
        
        self._init_weights()
    
    def _init_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.num_patches**0.5), cls_token=False
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, ids_restore):
        """
        x: visible patches from encoder [B, N_vis, D]
        ids_restore: indices to restore full sequence [B, N]
        """
        # Embed
        x = self.decoder_embed(x)  # [B, N_vis, decoder_dim]
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # [B, N, decoder_dim]
        
        # Unshuffle
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))
        
        # Add position embeddings
        x_ = x_ + self.decoder_pos_embed
        
        # Decoder blocks
        for blk in self.decoder_blocks:
            x_ = blk(x_)
        
        x_ = self.decoder_norm(x_)
        x_ = self.decoder_pred(x_)  # [B, N, patch_size^2 * 3]
        
        return x_


class MAEViTSmall(nn.Module):
    """
    MAE with ViT-Small encoder (matching JEPA architecture).
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 mask_ratio=0.75, decoder_embed_dim=192, decoder_depth=4):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # Encoder (matches JEPA's vit_small)
        self.encoder = vit_small()
        self.embed_dim = self.encoder.embed_dim  # 384
        
        # Patch embedding from encoder
        self.num_patches = (img_size // patch_size) ** 2  # 196 for 224/16
        
        # Decoder
        self.decoder = MAEDecoder(
            num_patches=self.num_patches,
            patch_size=patch_size,
            in_chans=in_chans,
            encoder_embed_dim=self.embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
        )
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first len_keep tokens
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def patchify(self, imgs):
        """Convert images to patches."""
        p = self.patch_size
        h = w = self.img_size // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * 3)
        return x
    
    def unpatchify(self, x):
        """Convert patches back to images."""
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * p, w * p)
        return imgs
    
    def forward_encoder(self, x, mask_ratio):
        """Encode with masking."""
        # Get patch embeddings
        x = self.encoder.patch_embed(x)  # [B, N, D]
        
        # Add position embeddings
        x = x + self.encoder.pos_embed
        
        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Apply transformer blocks
        for blk in self.encoder.blocks:
            x = blk(x)
        
        x = self.encoder.norm(x)
        
        return x, mask, ids_restore
    
    def forward_loss(self, imgs, pred, mask):
        """Compute reconstruction loss on masked patches."""
        target = self.patchify(imgs)
        
        # Normalize target
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5
        
        # MSE loss only on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N]
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        
        return loss, pred, mask


# ============================================================================
# Training
# ============================================================================

def train_mae(args):
    # Create output directory and setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = setup_logging(args.output_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Log file: {log_file}")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Save config
    config = vars(args)
    config['start_time'] = datetime.now().isoformat()
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Data transforms
    transform_list = [
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if args.horizontal_flip:
        transform_list.insert(1, transforms.RandomHorizontalFlip())
    transform = transforms.Compose(transform_list)
    
    # Load dataset
    filelist, frames, trace = load_dynamic_data(args.data_root)
    
    dataset = EchoPretrainDataset(
        root_path=args.data_root,
        filelist_df=filelist,
        frames_dict=frames,
        split="TRAIN",
        frame_type="both",
        transform=transform,
        img_size=args.img_size,
    )
    
    # Use num_workers=0 on Windows to avoid pickle issues
    num_workers = 0 if os.name == 'nt' else args.num_workers
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"Dataset: {len(dataset)} echo frames")
    logger.info(f"Batches per epoch: {len(dataloader)}")
    
    # Model
    model = MAEViTSmall(
        img_size=args.img_size,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        decoder_embed_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth
    ).to(device)
    
    logger.info(f"Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
    logger.info(f"Decoder parameters: {sum(p.numel() for p in model.decoder.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    # LR scheduler
    num_training_steps = args.epochs * len(dataloader)
    warmup_steps = args.warmup_epochs * len(dataloader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    all_losses = []
    best_loss = float('inf')
    grad_accum_steps = args.grad_accum_steps
    
    logger.info(f"Gradient accumulation steps: {grad_accum_steps}")
    logger.info(f"Effective batch size: {args.batch_size * grad_accum_steps}")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        optimizer.zero_grad()
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"{get_gpu_memory_info()}")
        logger.info(f"{'='*50}")
        
        try:
            for batch_idx, imgs in enumerate(dataloader):
                try:
                    # Sync before forward
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    imgs = imgs.to(device)
                    
                    loss, _, _ = model(imgs, args.mask_ratio)
                    loss = loss / grad_accum_steps
                    loss.backward()

                    # Sync after backward
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    # Step optimizer every grad_accum_steps
                    if (batch_idx + 1) % grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    actual_loss = loss.item() * grad_accum_steps
                    epoch_losses.append(actual_loss)
                    all_losses.append(actual_loss)
                    
                    if batch_idx % 25 == 0:  # Log more frequently
                        lr = optimizer.param_groups[0]['lr']
                        logger.info(f"  [{batch_idx}/{len(dataloader)}] Loss: {actual_loss:.4f} LR: {lr:.6f}")
                        logger.info(f"    {get_gpu_memory_info()}")
                        # Clear cache every log
                        clear_gpu_memory()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"CUDA OOM at batch {batch_idx}! {get_gpu_memory_info()}")
                        clear_gpu_memory()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise e
        
        except Exception as e:
            logger.error(f"Error during epoch {epoch+1}: {str(e)}")
            logger.error(f"{get_gpu_memory_info()}")
            # Save emergency checkpoint
            torch.save({
                'epoch': epoch,
                'encoder': model.encoder.state_dict(),
                'decoder': model.decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': np.mean(epoch_losses) if epoch_losses else 0,
            }, os.path.join(args.output_dir, f'emergency_checkpoint_ep{epoch}.pth'))
            raise e
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        logger.info(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
        
        # Clear memory at end of epoch
        clear_gpu_memory()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'encoder': model.encoder.state_dict(),
                'decoder': model.decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(args.output_dir, 'best_model.pth'))
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch + 1,
                'encoder': model.encoder.state_dict(),
                'decoder': model.decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(args.output_dir, f'checkpoint_ep{epoch+1}.pth'))
    
    # Save final encoder
    torch.save(model.encoder.state_dict(), os.path.join(args.output_dir, 'encoder_final.pth'))
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(all_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('MAE Training Loss')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'training_curve.png'), dpi=150)
    plt.close()
    
    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    logger.info(f"Model saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='MAE Pretraining on Echo')
    
    # Data
    parser.add_argument('--data_root', type=str,
                        default='Datasets/Echo/EchoNet_Dynamic')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)  # Further reduced to 4
    parser.add_argument('--num_workers', type=int, default=0)  # 0 for Windows
    parser.add_argument('--horizontal_flip', action='store_true', default=False)
    parser.add_argument('--grad_accum_steps', type=int, default=8)  # Effective batch = 4*8 = 32
    
    # Model
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--decoder_dim', type=int, default=192)
    parser.add_argument('--decoder_depth', type=int, default=4)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    
    # Output
    parser.add_argument('--output_dir', type=str,
                        default='experiments/echo_seg_pilot/mae')
    
    args = parser.parse_args()
    train_mae(args)


if __name__ == '__main__':
    main()
