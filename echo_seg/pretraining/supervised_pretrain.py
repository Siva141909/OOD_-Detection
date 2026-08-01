#!/usr/bin/env python3
"""
Supervised Pretraining Baseline on EchoNet-Dynamic

Uses ImageNet-pretrained ViT-Small as baseline.
No actual training needed - just converts weights to match JEPA encoder format.

This creates a comparable baseline that:
1. Uses SAME architecture as JEPA/MAE (vit_small)
2. Uses ImageNet-pretrained weights from timm
3. Can be directly compared in downstream tasks
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime

import torch
import timm

# Add IJEPA path
IJEPA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "IJEPA_Meta")
sys.path.insert(0, IJEPA_PATH)
from src.models.vision_transformer import vit_small

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def convert_timm_to_jepa(timm_state_dict, jepa_model):
    """
    Convert timm ViT weights to JEPA vit_small format.
    
    Both use patch_size=16, embed_dim=384, depth=12, num_heads=6.
    Main differences are in naming conventions.
    """
    jepa_state = jepa_model.state_dict()
    new_state = {}
    
    # Mapping from timm to JEPA naming
    for key, value in timm_state_dict.items():
        new_key = key
        
        # Handle patch embedding
        if key == "patch_embed.proj.weight":
            new_key = "patch_embed.proj.weight"
        elif key == "patch_embed.proj.bias":
            new_key = "patch_embed.proj.bias"
        
        # Handle position embedding
        elif key == "pos_embed":
            # JEPA doesn't include [CLS] token in pos_embed
            # timm shape: [1, 197, 384], JEPA shape: [1, 196, 384]
            if value.shape[1] == 197:
                new_key = "pos_embed"
                value = value[:, 1:, :]  # Remove CLS position
            else:
                new_key = "pos_embed"
        
        # Handle CLS token - JEPA doesn't use it in the same way
        elif key == "cls_token":
            continue  # Skip CLS token
        
        # Handle blocks
        elif key.startswith("blocks."):
            new_key = key  # Same naming for blocks
        
        # Handle norm
        elif key in ["norm.weight", "norm.bias"]:
            new_key = key
        
        # Handle head (not used in encoder-only mode)
        elif key.startswith("head"):
            continue
        
        # Check if key exists in JEPA model
        if new_key in jepa_state:
            if jepa_state[new_key].shape == value.shape:
                new_state[new_key] = value
            else:
                logger.warning(f"Shape mismatch for {new_key}: "
                             f"JEPA={jepa_state[new_key].shape}, timm={value.shape}")
        else:
            logger.debug(f"Key {new_key} not found in JEPA model")
    
    return new_state


def create_supervised_baseline(args):
    """Create supervised baseline from ImageNet pretrained weights."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    config = vars(args)
    config['creation_time'] = datetime.now().isoformat()
    config['description'] = 'ImageNet-pretrained ViT-Small baseline'
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load ImageNet pretrained ViT-Small from timm
    logger.info("Loading ImageNet-pretrained ViT-Small from timm...")
    timm_model = timm.create_model('vit_small_patch16_224', pretrained=True)
    timm_state = timm_model.state_dict()
    
    # Create JEPA vit_small
    logger.info("Creating JEPA vit_small model...")
    jepa_model = vit_small()
    
    # Convert weights
    logger.info("Converting timm weights to JEPA format...")
    converted_state = convert_timm_to_jepa(timm_state, jepa_model)
    
    # Load converted weights
    missing, unexpected = jepa_model.load_state_dict(converted_state, strict=False)
    logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if missing:
        logger.info(f"Missing: {missing[:10]}...")  # Show first 10
    
    # Save in same format as JEPA/MAE
    save_path = os.path.join(args.output_dir, 'encoder_final.pth')
    torch.save(jepa_model.state_dict(), save_path)
    logger.info(f"Saved encoder to {save_path}")
    
    # Also save with metadata
    checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
    torch.save({
        'epoch': 0,
        'encoder': jepa_model.state_dict(),
        'source': 'imagenet_pretrained',
        'timm_model': 'vit_small_patch16_224',
    }, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Verify model works
    logger.info("Verifying model...")
    jepa_model.to(device)
    jepa_model.eval()
    
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        out = jepa_model(dummy)
        logger.info(f"Output shape: {out.shape}")  # Should be [1, 196, 384]
    
    logger.info("Supervised baseline created successfully!")
    
    # Write summary
    summary = {
        'model': 'vit_small',
        'source': 'ImageNet-1k pretrained (timm)',
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'patch_size': 16,
        'num_params': sum(p.numel() for p in jepa_model.parameters()),
        'output_shape': list(out.shape),
        'weights_converted': len(converted_state),
        'missing_keys': len(missing),
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return jepa_model


def main():
    parser = argparse.ArgumentParser(description='Create Supervised Baseline')
    parser.add_argument('--output_dir', type=str,
                        default='experiments/echo_seg_pilot/supervised')
    
    args = parser.parse_args()
    create_supervised_baseline(args)


if __name__ == '__main__':
    main()
