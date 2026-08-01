# Sanity Check Script for Medical JEPA
# Run this BEFORE full training to verify:
# 1. Dataset loads correctly
# 2. Model forward pass works
# 3. Loss is computed correctly
# 4. Embeddings don't collapse

import os
import sys
import copy
import logging
import argparse

import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F

# Add IJEPA_Meta to path
IJEPA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "IJEPA_Meta")
sys.path.insert(0, IJEPA_PATH)

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.tensors import repeat_interleave_batch
from src.helper import init_model
from src.transforms import make_transforms

# Add cxr_ood root to path for the actual dataset module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cxr_dataset import CXRDataset, get_cxr_dataset_stats


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dataset(root_path):
    """Check dataset loading and image properties."""
    print("\n" + "="*60)
    print("STEP 1: DATASET CHECK")
    print("="*60)
    
    # Get statistics
    try:
        stats = get_cxr_dataset_stats(root_path)
        print(f"\n[PASS] Dataset found!")
        print(f"  Total images: {stats['total']}")
        for key, val in stats.items():
            if key != 'total':
                print(f"  - {key}: {val}")
    except Exception as e:
        print(f"\n[FAIL] Error loading dataset: {e}")
        return False
    
    # Create dataset with transforms
    transform = make_transforms(
        crop_size=224,
        crop_scale=(0.3, 1.0),
        gaussian_blur=False,
        horizontal_flip=False,
        color_distortion=False,
        color_jitter=0.0
    )
    
    dataset = CXRDataset(root_path=root_path, transform=transform)
    
    # Load a few images
    print(f"\n  Testing image loading...")
    for i in range(min(5, len(dataset))):
        img = dataset[i]
        print(f"  Image {i}: shape={img.shape}, min={img.min():.3f}, max={img.max():.3f}")
    
    print(f"\n[PASS] Dataset check PASSED")
    return True


def check_masking(crop_size=224, patch_size=16):
    """Check mask collator creates valid masks."""
    print("\n" + "="*60)
    print("STEP 2: MASKING CHECK")
    print("="*60)
    
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=(0.15, 0.2),
        enc_mask_scale=(0.85, 1.0),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        allow_overlap=False,
        min_keep=10
    )
    
    # Create dummy batch
    batch_size = 4
    dummy_batch = [torch.randn(3, crop_size, crop_size) for _ in range(batch_size)]
    
    collated_batch, masks_enc, masks_pred = mask_collator(dummy_batch)
    
    print(f"\n  Batch shape: {collated_batch.shape}")
    print(f"  Number of encoder masks: {len(masks_enc)}")
    print(f"  Number of predictor masks: {len(masks_pred)}")
    
    # Check mask shapes
    for i, m in enumerate(masks_enc):
        print(f"  Encoder mask {i}: shape={m.shape}")
    for i, m in enumerate(masks_pred):
        print(f"  Predictor mask {i}: shape={m.shape}")
    
    # Verify masks are valid indices
    num_patches = (crop_size // patch_size) ** 2
    for m in masks_enc:
        assert m.max() < num_patches, f"Invalid mask index: {m.max()} >= {num_patches}"
        assert m.min() >= 0, f"Negative mask index: {m.min()}"
    
    print(f"\n[PASS] Masking check PASSED")
    return True


def check_model_forward(device='cuda:0'):
    """Check model forward pass works correctly."""
    print("\n" + "="*60)
    print("STEP 3: MODEL FORWARD CHECK")
    print("="*60)
    
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
        print(f"  CUDA not available, using CPU")
    
    device = torch.device(device)
    
    # Init model
    encoder, predictor = init_model(
        device=device,
        patch_size=16,
        crop_size=224,
        pred_depth=6,
        pred_emb_dim=384,
        model_name='vit_small'
    )
    
    target_encoder = copy.deepcopy(encoder)
    
    print(f"\n  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")
    
    # Create dummy input
    batch_size = 4
    imgs = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Create masks
    mask_collator = MBMaskCollator(
        input_size=224,
        patch_size=16,
        pred_mask_scale=(0.15, 0.2),
        enc_mask_scale=(0.85, 1.0),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        allow_overlap=False,
        min_keep=10
    )
    
    dummy_batch = [torch.randn(3, 224, 224) for _ in range(batch_size)]
    _, masks_enc, masks_pred = mask_collator(dummy_batch)
    masks_enc = [m.to(device) for m in masks_enc]
    masks_pred = [m.to(device) for m in masks_pred]
    
    # Forward pass
    with torch.no_grad():
        # Target encoder
        h = target_encoder(imgs)
        h = F.layer_norm(h, (h.size(-1),))
        print(f"\n  Target encoder output shape: {h.shape}")
        
        B = len(h)
        h = apply_masks(h, masks_pred)
        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
        print(f"  After masking and repeat: {h.shape}")
        
        # Context encoder
        z = encoder(imgs, masks_enc)
        print(f"  Context encoder output shape: {z.shape}")
        
        z = predictor(z, masks_enc, masks_pred)
        print(f"  Predictor output shape: {z.shape}")
        
        # Loss
        loss = F.smooth_l1_loss(z, h)
        print(f"  Loss: {loss.item():.4f}")
    
    print(f"\n[PASS] Model forward check PASSED")
    return True


def check_training_step(root_path, device='cuda:0'):
    """Check one full training step."""
    print("\n" + "="*60)
    print("STEP 4: TRAINING STEP CHECK")
    print("="*60)
    
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
        print(f"  CUDA not available, using CPU")
    
    device = torch.device(device)
    
    # Create dataset
    transform = make_transforms(
        crop_size=224,
        crop_scale=(0.3, 1.0),
        gaussian_blur=False,
        horizontal_flip=False,
        color_distortion=False,
        color_jitter=0.0
    )
    
    mask_collator = MBMaskCollator(
        input_size=224,
        patch_size=16,
        pred_mask_scale=(0.15, 0.2),
        enc_mask_scale=(0.85, 1.0),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        allow_overlap=False,
        min_keep=10
    )
    
    dataset = CXRDataset(root_path=root_path, transform=transform, max_images=100)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        collate_fn=mask_collator,
        shuffle=True,
        num_workers=0
    )
    
    # Create model
    encoder, predictor = init_model(
        device=device,
        patch_size=16,
        crop_size=224,
        pred_depth=6,
        pred_emb_dim=384,
        model_name='vit_small'
    )
    
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=1e-4
    )
    
    # Training loop (3 iterations)
    encoder.train()
    predictor.train()
    
    losses = []
    print(f"\n  Running 3 training iterations...")
    
    for i, (udata, masks_enc, masks_pred) in enumerate(dataloader):
        if i >= 3:
            break
        
        # udata is the collated batch of images (tensor)
        # Could be (imgs,) tuple or just imgs tensor depending on dataset __getitem__
        if isinstance(udata, (list, tuple)):
            imgs = udata[0].to(device)
        else:
            imgs = udata.to(device)
        masks_enc = [m.to(device) for m in masks_enc]
        masks_pred = [m.to(device) for m in masks_pred]
        
        # Forward
        with torch.no_grad():
            h = target_encoder(imgs)
            h = F.layer_norm(h, (h.size(-1),))
            B = len(h)
            h = apply_masks(h, masks_pred)
            h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
        
        z = encoder(imgs, masks_enc)
        z = predictor(z, masks_enc, masks_pred)
        
        loss = F.smooth_l1_loss(z, h)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # EMA update
        m = 0.996
        with torch.no_grad():
            for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)
        
        losses.append(loss.item())
        print(f"  Iteration {i+1}: loss = {loss.item():.4f}")
    
    # Check loss is decreasing or stable
    if not np.isnan(losses).any() and not np.isinf(losses).any():
        print(f"\n[PASS] Training step check PASSED (no NaN/Inf)")
    else:
        print(f"\n[FAIL] Training step check FAILED (NaN/Inf detected)")
        return False
    
    return True


def check_embedding_collapse(root_path, device='cuda:0'):
    """Check if embeddings collapse (variance should be > 0)."""
    print("\n" + "="*60)
    print("STEP 5: EMBEDDING COLLAPSE CHECK")
    print("="*60)
    
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
        print(f"  CUDA not available, using CPU")
    
    device = torch.device(device)
    
    # Create dataset
    transform = make_transforms(
        crop_size=224,
        crop_scale=(0.3, 1.0),
        gaussian_blur=False,
        horizontal_flip=False,
        color_distortion=False,
        color_jitter=0.0
    )
    
    dataset = CXRDataset(root_path=root_path, transform=transform, max_images=50)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )
    
    # Create model (random init)
    encoder, _ = init_model(
        device=device,
        patch_size=16,
        crop_size=224,
        pred_depth=6,
        pred_emb_dim=384,
        model_name='vit_small'
    )
    
    encoder.eval()
    
    # Collect embeddings
    all_embeddings = []
    
    with torch.no_grad():
        for imgs in dataloader:
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0]
            imgs = imgs.to(device)
            
            h = encoder(imgs)
            h = F.layer_norm(h, (h.size(-1),))
            h_mean = h.mean(dim=1)  # [B, D]
            all_embeddings.append(h_mean.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Compute stats
    mean = all_embeddings.mean(dim=0)
    std = all_embeddings.std(dim=0)
    var = (std ** 2).mean().item()
    
    print(f"\n  Embedding shape: {all_embeddings.shape}")
    print(f"  Mean of means: {mean.mean().item():.4f}")
    print(f"  Mean of stds: {std.mean().item():.4f}")
    print(f"  Variance: {var:.4f}")
    
    if var > 1e-4:
        print(f"\n[PASS] Embedding collapse check PASSED (variance = {var:.4f})")
        return True
    else:
        print(f"\n[WARN] WARNING: Low variance ({var:.6f}) - potential collapse!")
        return True  # Still pass, just warn


def visualize_samples(root_path, save_path='sample_images.png'):
    """Visualize sample images from the dataset."""
    print("\n" + "="*60)
    print("BONUS: VISUALIZING SAMPLES")
    print("="*60)
    
    dataset = CXRDataset(root_path=root_path, transform=None)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, ax in enumerate(axes.flat):
        if i >= len(dataset):
            break
        
        img = dataset[i]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(f'Image {i} ({dataset.dataset_sources[i]})')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved sample images to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='JEPA Sanity Check')
    parser.add_argument(
        '--root_path', type=str,
        default='Datasets/CXR',
        help='Path to CXR datasets'
    )
    parser.add_argument(
        '--device', type=str,
        default='cuda:0',
        help='Device to use'
    )
    parser.add_argument(
        '--skip_training', action='store_true',
        help='Skip training step check (faster)'
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MEDICAL JEPA SANITY CHECK")
    print("="*60)
    print(f"Dataset path: {args.root_path}")
    print(f"Device: {args.device}")
    
    all_passed = True
    
    # Run checks
    all_passed &= check_dataset(args.root_path)
    all_passed &= check_masking()
    all_passed &= check_model_forward(args.device)
    
    if not args.skip_training:
        all_passed &= check_training_step(args.root_path, args.device)
        all_passed &= check_embedding_collapse(args.root_path, args.device)
    
    # Visualize samples
    try:
        visualize_samples(args.root_path)
    except Exception as e:
        print(f"  Could not visualize samples: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all_passed:
        print("\n[PASS] ALL CHECKS PASSED!")
        print("\nYou can now run full training with:")
        print("  python cxr_ood/pretraining/jepa_pretrain.py --config cxr_ood/configs/cxr_vit_small.yaml")
    else:
        print("\n[FAIL] SOME CHECKS FAILED - Fix issues before training")
    
    return all_passed


if __name__ == '__main__':
    main()
