#!/usr/bin/env python3
"""
Linear Probe Evaluation for JEPA Embeddings

Evaluates JEPA encoder quality by:
1. Freezing the JEPA encoder
2. Training a single linear classification head
3. Evaluating on in-domain data (Normal vs TB classification)

Metrics:
- AUROC
- Accuracy  
- ECE (Expected Calibration Error)
- NLL (Negative Log-Likelihood)
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
import yaml
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Windows
import matplotlib.pyplot as plt
from datetime import datetime

# Add IJEPA_Meta to path
IJEPA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "IJEPA_Meta")
sys.path.insert(0, IJEPA_PATH)
from src.models.vision_transformer import vit_small, vit_base, vit_large, vit_huge

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset for Classification (with labels)
# ============================================================================

class TBClassificationDataset(Dataset):
    """
    TB Chest Radiography Database with labels for classification.
    Labels: 0 = Normal, 1 = Tuberculosis
    """
    
    SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        # Collect Normal images (label=0)
        normal_folder = os.path.join(root_path, 'Normal')
        if os.path.exists(normal_folder):
            for f in os.listdir(normal_folder):
                if f.endswith(self.SUPPORTED_EXTENSIONS):
                    self.image_paths.append(os.path.join(normal_folder, f))
                    self.labels.append(0)
        
        # Collect Tuberculosis images (label=1)
        tb_folder = os.path.join(root_path, 'Tuberculosis')
        if os.path.exists(tb_folder):
            for f in os.listdir(tb_folder):
                if f.endswith(self.SUPPORTED_EXTENSIONS):
                    self.image_paths.append(os.path.join(tb_folder, f))
                    self.labels.append(1)
        
        logger.info(f"Loaded {len(self.image_paths)} images: "
                   f"{self.labels.count(0)} Normal, {self.labels.count(1)} TB")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            # Return a placeholder
            img = Image.new('RGB', (224, 224), (128, 128, 128))
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label


# ============================================================================
# Linear Probe Classifier
# ============================================================================

class LinearProbe(nn.Module):
    """
    Frozen encoder + trainable linear head for classification.
    """
    
    def __init__(self, encoder, embed_dim, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
    
    def forward(self, x):
        # This backbone has no CLS token (cls_token=False in IJEPA's ViT);
        # mean-pool over patch tokens for a global representation.
        with torch.no_grad():
            features = self.encoder(x)  # [B, N, D]
            pooled = features.mean(dim=1)  # [B, D]

        logits = self.head(pooled)
        return logits

    def get_embeddings(self, x):
        """Extract embeddings without classification."""
        with torch.no_grad():
            features = self.encoder(x)
            return features.mean(dim=1)


# ============================================================================
# Calibration Metrics
# ============================================================================

def compute_ece(probs, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).
    
    ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|
    
    Args:
        probs: [N, C] predicted probabilities
        labels: [N] ground truth labels
        n_bins: number of confidence bins
    
    Returns:
        ECE value
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Get confidence (max probability) and predictions
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return ece


def compute_nll(logits, labels):
    """
    Compute Negative Log-Likelihood (cross-entropy loss).
    
    Args:
        logits: [N, C] model logits
        labels: [N] ground truth labels
    
    Returns:
        NLL value (per sample)
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    
    nll = F.cross_entropy(logits, labels, reduction='mean')
    return nll.item()


# ============================================================================
# Load JEPA Checkpoint
# ============================================================================

def load_jepa_encoder(checkpoint_path, model_name='vit_small', device='cuda'):
    """
    Load pretrained JEPA encoder from checkpoint.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model architecture
    model_fn = {
        'vit_small': vit_small,
        'vit_base': vit_base,
        'vit_large': vit_large,
        'vit_huge': vit_huge
    }[model_name]
    
    # Create encoder
    encoder = model_fn()
    
    # Load state dict (target encoder is the EMA-updated one)
    if 'target_encoder' in checkpoint:
        state_dict = checkpoint['target_encoder']
        logger.info("Loading target (EMA) encoder weights")
    elif 'encoder' in checkpoint:
        state_dict = checkpoint['encoder']
        logger.info("Loading encoder weights")
    else:
        raise ValueError("No encoder weights found in checkpoint")
    
    # Load weights
    encoder.load_state_dict(state_dict)
    encoder = encoder.to(device)
    encoder.eval()
    
    # Get embedding dimension
    embed_dim = encoder.embed_dim
    
    logger.info(f"Loaded {model_name} encoder with embed_dim={embed_dim}")
    return encoder, embed_dim


# ============================================================================
# Training History Tracker
# ============================================================================

class TrainingHistory:
    """
    Track and save training/validation metrics over epochs.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Training metrics
        self.train_loss = []
        self.train_acc = []
        self.learning_rates = []
        
        # Validation metrics (recorded every epoch)
        self.val_auroc = []
        self.val_accuracy = []
        self.val_ece = []
        self.val_nll = []
        
        # Best metrics
        self.best_epoch = 0
        self.best_auroc = 0.0
        
        # Timestamps
        self.start_time = datetime.now()
        self.epoch_times = []
    
    def log_train_epoch(self, epoch, loss, acc, lr):
        """Log training metrics for an epoch."""
        self.train_loss.append(loss)
        self.train_acc.append(acc)
        self.learning_rates.append(lr)
    
    def log_val_epoch(self, epoch, metrics):
        """Log validation metrics for an epoch."""
        self.val_auroc.append(metrics['auroc'])
        self.val_accuracy.append(metrics['accuracy'])
        self.val_ece.append(metrics['ece'])
        self.val_nll.append(metrics['nll'])
        
        if metrics['auroc'] > self.best_auroc:
            self.best_auroc = metrics['auroc']
            self.best_epoch = epoch
    
    def save_history(self):
        """Save training history to JSON file."""
        history = {
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'learning_rates': self.learning_rates,
            'val_auroc': self.val_auroc,
            'val_accuracy': self.val_accuracy,
            'val_ece': self.val_ece,
            'val_nll': self.val_nll,
            'best_epoch': self.best_epoch,
            'best_auroc': self.best_auroc,
            'training_time_seconds': (datetime.now() - self.start_time).total_seconds()
        }
        
        path = os.path.join(self.output_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to {path}")
        return history
    
    def plot_curves(self):
        """Plot and save training curves."""
        epochs = range(1, len(self.train_loss) + 1)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Linear Probe Training Progress', fontsize=14, fontweight='bold')
        
        # 1. Training Loss
        ax = axes[0, 0]
        ax.plot(epochs, self.train_loss, 'b-', linewidth=2, label='Train Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Training Accuracy
        ax = axes[0, 1]
        ax.plot(epochs, self.train_acc, 'g-', linewidth=2, label='Train Acc')
        if self.val_accuracy:
            ax.plot(epochs, self.val_accuracy, 'r-', linewidth=2, label='Val Acc')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. Validation AUROC
        ax = axes[0, 2]
        if self.val_auroc:
            ax.plot(epochs, self.val_auroc, 'purple', linewidth=2, label='Val AUROC')
            ax.axvline(x=self.best_epoch + 1, color='red', linestyle='--', alpha=0.7, label=f'Best (ep{self.best_epoch+1})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUROC')
        ax.set_title('Validation AUROC')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 4. Validation ECE
        ax = axes[1, 0]
        if self.val_ece:
            ax.plot(epochs, self.val_ece, 'orange', linewidth=2, label='Val ECE')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('ECE')
        ax.set_title('Calibration Error (ECE) - Lower is Better')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 5. Validation NLL
        ax = axes[1, 1]
        if self.val_nll:
            ax.plot(epochs, self.val_nll, 'brown', linewidth=2, label='Val NLL')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('NLL')
        ax.set_title('Negative Log-Likelihood - Lower is Better')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 6. Learning Rate
        ax = axes[1, 2]
        ax.plot(epochs, self.learning_rates, 'teal', linewidth=2, label='LR')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves saved to {path}")


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_linear_probe(model, train_loader, val_loader, device, 
                       epochs=50, lr=0.001, weight_decay=0.0, output_dir='./results'):
    """
    Train the linear classification head with full logging.
    """
    # Initialize history tracker
    history = TrainingHistory(output_dir)
    
    # Only train the head parameters
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_auroc = 0.0
    best_state = None
    
    # Print header
    print("\n" + "="*100)
    print(f"{'Epoch':^8} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val AUROC':^10} | {'Val Acc':^10} | {'Val ECE':^10} | {'Val NLL':^10} | {'LR':^12}")
    print("="*100)
    
    for epoch in range(epochs):
        model.head.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            num_batches += 1
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        train_acc = correct / total
        
        # Validate every epoch
        val_metrics = evaluate_model(model, val_loader, device)
        
        # Log to history
        history.log_train_epoch(epoch, avg_loss, train_acc, current_lr)
        history.log_val_epoch(epoch, val_metrics)
        
        # Track best model
        is_best = ""
        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            best_state = {k: v.clone() for k, v in model.head.state_dict().items()}
            is_best = " *BEST*"
            
            # Save best model checkpoint immediately
            best_model_path = os.path.join(output_dir, 'best_linear_probe.pth')
            torch.save({
                'epoch': epoch + 1,
                'head_state_dict': best_state,
                'embed_dim': model.embed_dim,
                'num_classes': 2,
                'val_auroc': best_auroc,
                'val_accuracy': val_metrics['accuracy'],
                'val_ece': val_metrics['ece'],
                'val_nll': val_metrics['nll'],
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, best_model_path)
        
        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'head_state_dict': model.head.state_dict(),
                'embed_dim': model.embed_dim,
                'num_classes': 2,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_loss,
                'val_metrics': val_metrics,
            }, ckpt_path)
            logger.info(f"Checkpoint saved to {ckpt_path}")
        
        # Print progress
        print(f"{epoch+1:^8} | {avg_loss:^12.4f} | {train_acc:^10.4f} | {val_metrics['auroc']:^10.4f} | {val_metrics['accuracy']:^10.4f} | {val_metrics['ece']:^10.4f} | {val_metrics['nll']:^10.4f} | {current_lr:^12.2e}{is_best}")
    
    print("="*100)
    print(f"Best validation AUROC: {best_auroc:.4f} at epoch {history.best_epoch + 1}")
    print("="*100 + "\n")
    
    # Restore best model
    if best_state is not None:
        model.head.load_state_dict(best_state)
        logger.info(f"Restored best model from epoch {history.best_epoch + 1}")
    
    # Save history and plots
    history.save_history()
    history.plot_curves()
    
    return model, history


def evaluate_model(model, data_loader, device):
    """
    Evaluate model and compute all metrics.
    """
    model.eval()
    
    all_logits = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            
            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Predictions
    predictions = all_probs.argmax(dim=1).numpy()
    labels_np = all_labels.numpy()
    probs_np = all_probs.numpy()
    
    # Compute metrics
    # AUROC - use probability of positive class (TB)
    auroc = roc_auc_score(labels_np, probs_np[:, 1])
    
    # Accuracy
    accuracy = accuracy_score(labels_np, predictions)
    
    # ECE
    ece = compute_ece(probs_np, labels_np)
    
    # NLL
    nll = compute_nll(all_logits, all_labels)
    
    return {
        'auroc': auroc,
        'accuracy': accuracy,
        'ece': ece,
        'nll': nll,
        'n_samples': len(labels_np)
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Linear Probe Evaluation for JEPA')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to JEPA checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to TB_Chest_Radiography_Database')
    parser.add_argument('--model-name', type=str, default='vit_small',
                       choices=['vit_small', 'vit_base', 'vit_large', 'vit_huge'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Fraction of data for training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to pre-trained linear probe (skip training, only evaluate)')
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========================================================================
    # 1. Load JEPA encoder (frozen)
    # ========================================================================
    encoder, embed_dim = load_jepa_encoder(
        args.checkpoint, 
        model_name=args.model_name,
        device=device
    )
    
    # ========================================================================
    # 2. Create linear probe model
    # ========================================================================
    model = LinearProbe(encoder, embed_dim, num_classes=2).to(device)
    logger.info(f"Linear probe created with {sum(p.numel() for p in model.head.parameters())} trainable parameters")
    
    # ========================================================================
    # 3. Prepare dataset
    # ========================================================================
    # Simple transforms (no augmentation for evaluation)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = TBClassificationDataset(args.data_path, transform=transform)
    
    # Split into train/val
    n_total = len(full_dataset)
    n_train = int(n_total * args.train_split)
    n_val = n_total - n_train
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Save dataset split indices for reproducibility
    split_info = {
        'n_total': n_total,
        'n_train': n_train,
        'n_val': n_val,
        'train_indices': train_dataset.indices,
        'val_indices': val_dataset.indices,
        'seed': args.seed,
        'data_path': args.data_path
    }
    split_path = os.path.join(args.output_dir, 'dataset_split.json')
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"Dataset split saved to {split_path}")
    
    logger.info(f"Dataset split: {n_train} train, {n_val} val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # ========================================================================
    # 4. Train or Load linear probe
    # ========================================================================
    if args.load_model is not None:
        # Load pre-trained linear probe
        logger.info(f"Loading pre-trained linear probe from {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=device)
        model.head.load_state_dict(checkpoint['head_state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_auroc' in checkpoint:
            logger.info(f"Model was trained with val AUROC: {checkpoint['val_auroc']:.4f}")
        history = None  # No training history when loading
    else:
        # Train from scratch
        logger.info("Training linear probe...")
        model, history = train_linear_probe(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, output_dir=args.output_dir
        )
    
    # ========================================================================
    # 5. Final evaluation
    # ========================================================================
    logger.info("Final evaluation on validation set...")
    final_metrics = evaluate_model(model, val_loader, device)
    
    # ========================================================================
    # 6. Print results table
    # ========================================================================
    print("\n" + "="*60)
    print("JEPA Linear Probe - In-Domain Performance")
    print("="*60)
    print(f"Dataset: TB_Chest_Radiography_Database (Normal vs TB)")
    print(f"Model: {args.model_name} (frozen encoder)")
    print(f"Train/Val split: {n_train}/{n_val}")
    print("-"*60)
    print(f"{'Metric':<20} {'Value':<15}")
    print("-"*60)
    print(f"{'AUROC':<20} {final_metrics['auroc']:.4f}")
    print(f"{'Accuracy':<20} {final_metrics['accuracy']:.4f}")
    print(f"{'ECE':<20} {final_metrics['ece']:.4f}")
    print(f"{'NLL':<20} {final_metrics['nll']:.4f}")
    print("="*60)
    
    # Save results
    results = {
        'model': args.model_name,
        'checkpoint': args.checkpoint,
        'dataset': 'TB_Chest_Radiography_Database',
        'n_train': n_train,
        'n_val': n_val,
        'metrics': final_metrics,
        'best_epoch': history.best_epoch + 1 if history else 'loaded',
        'training_epochs': args.epochs,
        'learning_rate': args.lr,
        'batch_size': args.batch_size
    }
    
    results_path = os.path.join(args.output_dir, 'linear_probe_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    logger.info(f"Results saved to {results_path}")
    
    # Save model (only if we trained, not if we loaded)
    if history is not None:
        model_path = os.path.join(args.output_dir, 'linear_probe_model.pth')
        torch.save({
            'head_state_dict': model.head.state_dict(),
            'embed_dim': embed_dim,
            'num_classes': 2,
            'model_name': args.model_name,
            'jepa_checkpoint': args.checkpoint,
            'final_metrics': final_metrics,
            'best_epoch': history.best_epoch + 1,
            'training_config': {
                'epochs': args.epochs,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'train_split': args.train_split,
                'seed': args.seed,
            }
        }, model_path)
        logger.info(f"Final model saved to {model_path}")
    else:
        logger.info("Skipped model saving (loaded pre-trained model)")
    
    return final_metrics


if __name__ == '__main__':
    main()
