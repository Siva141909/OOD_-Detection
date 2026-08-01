#!/usr/bin/env python3
"""
Baseline Comparison: JEPA vs MAE vs Supervised ImageNet

Runs the complete evaluation pipeline for each encoder:
1. Linear probe training on ID data
2. Calibration analysis (ECE)
3. OOD detection (Energy, Mahalanobis)

Outputs a unified comparison table.
"""

import os
import sys
import argparse
import json
import logging
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.covariance import EmpiricalCovariance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add IJEPA_Meta to path for JEPA models
IJEPA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "IJEPA_Meta")
sys.path.insert(0, IJEPA_PATH)
from src.models.vision_transformer import vit_small, vit_base

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Datasets
# ============================================================================

class TBClassificationDataset(Dataset):
    """TB Chest Radiography Database with labels."""
    SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label, folder_name in [(0, 'Normal'), (1, 'Tuberculosis')]:
            folder = os.path.join(root_path, folder_name)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.endswith(self.SUPPORTED_EXTENSIONS):
                        self.image_paths.append(os.path.join(folder, f))
                        self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class MontgomeryDataset(Dataset):
    """Montgomery TB CXR Dataset."""
    SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    
    def __init__(self, root_path, transform=None):
        self.transform = transform
        self.image_paths = []
        folder = os.path.join(root_path, 'Montgomery TB CXR', 'images')
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(self.SUPPORTED_EXTENSIONS):
                    self.image_paths.append(os.path.join(folder, f))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, -1


class ShenzhenDataset(Dataset):
    """Shenzhen TB CXR Dataset."""
    SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    
    def __init__(self, root_path, transform=None):
        self.transform = transform
        self.image_paths = []
        folder = os.path.join(root_path, 'Shenzhen TB CXR', 'images', 'images')
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(self.SUPPORTED_EXTENSIONS):
                    self.image_paths.append(os.path.join(folder, f))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, -1


# ============================================================================
# Encoder Wrappers
# ============================================================================

class JEPAEncoder(nn.Module):
    """Wrapper for JEPA-pretrained ViT."""
    def __init__(self, checkpoint_path, model_name='vit_small', device='cuda'):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        model_fn = {'vit_small': vit_small, 'vit_base': vit_base}[model_name]
        self.encoder = model_fn()
        
        if 'target_encoder' in checkpoint:
            self.encoder.load_state_dict(checkpoint['target_encoder'])
        else:
            self.encoder.load_state_dict(checkpoint['encoder'])
        
        self.embed_dim = self.encoder.embed_dim
        self.encoder.eval()
        
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)  # [B, N, D] - no CLS token
            return features.mean(dim=1)  # Mean pool over patches


class SupervisedEncoder(nn.Module):
    """
    Wrapper for Supervised baseline using ViT-Small pretrained on TB CXR.
    Loads pretrained checkpoint for fair comparison (same data, same arch, different method).
    """
    def __init__(self, checkpoint_path=None, device='cuda'):
        super().__init__()
        self.encoder = vit_small()
        self.embed_dim = self.encoder.embed_dim  # 384, same as JEPA
        
        # Load pretrained supervised checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading Supervised checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # The supervised checkpoint saves encoder directly in 'encoder' key
            if 'encoder' in checkpoint:
                encoder_state = checkpoint['encoder']
                self.encoder.load_state_dict(encoder_state, strict=True)
                logger.info(f"Loaded Supervised encoder weights ({len(encoder_state)} params)")
            elif 'model_state_dict' in checkpoint:
                # Alternative format: filter encoder weights from full model
                state_dict = checkpoint['model_state_dict']
                encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() 
                               if k.startswith('encoder.') and 'head' not in k}
                if encoder_state:
                    self.encoder.load_state_dict(encoder_state, strict=False)
                    logger.info(f"Loaded Supervised encoder weights ({len(encoder_state)} params)")
            else:
                logger.warning("No encoder weights found in checkpoint, using random init")
        else:
            logger.warning(f"No Supervised checkpoint provided or not found, using random init")
        
        logger.info(f"Initialized Supervised encoder: ViT-Small (embed_dim={self.embed_dim})")
        
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)  # [B, N, D] - no CLS token
            return features.mean(dim=1)  # Mean pool over patches


class MAEEncoder(nn.Module):
    """
    Wrapper for MAE baseline pretrained on TB CXR.
    Uses vit_small() - same architecture as JEPA and Supervised.
    """
    def __init__(self, checkpoint_path=None, device='cuda'):
        super().__init__()
        
        # Use the same vit_small as JEPA and Supervised
        self.encoder = vit_small(patch_size=16, drop_path_rate=0.1)
        self.embed_dim = self.encoder.embed_dim
        
        # Load pretrained MAE encoder checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading MAE checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # MAE v2 checkpoint saves encoder in 'encoder' key
            if 'encoder' in checkpoint:
                encoder_state = checkpoint['encoder']
                # Load with strict=True - should match exactly now
                self.encoder.load_state_dict(encoder_state, strict=True)
                logger.info(f"Loaded MAE encoder weights ({len(encoder_state)} keys)")
            else:
                logger.warning("No 'encoder' key found in checkpoint, using random init")
        else:
            logger.warning(f"No MAE checkpoint provided or not found, using random init")
        
        logger.info(f"Initialized MAE encoder: ViT-Small (embed_dim={self.embed_dim})")
        
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)  # [B, N, D] - no CLS token
            return features.mean(dim=1)  # Mean pool over patches


# ============================================================================
# Linear Probe
# ============================================================================

class LinearProbe(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)
    
    def get_embeddings(self, x):
        return self.encoder(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_ece(probs, labels, n_bins=15):
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            ece += np.abs(avg_acc - avg_conf) * in_bin.mean()
    return ece


def compute_nll(logits, labels):
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    return F.cross_entropy(logits, labels, reduction='mean').item()


def compute_ood_auroc(id_scores, ood_scores):
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    return roc_auc_score(labels, scores)


def compute_fpr_at_tpr(id_scores, ood_scores, tpr_threshold=0.95):
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmax(tpr >= tpr_threshold)
    return fpr[idx]


def compute_energy_score(logits, temperature=1.0):
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


class MahalanobisDetector:
    def __init__(self):
        self.class_means = {}
        self.precision = None
    
    def fit(self, embeddings, labels):
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            self.class_means[label] = embeddings[mask].mean(axis=0)
        
        centered = []
        for label in unique_labels:
            mask = labels == label
            centered.append(embeddings[mask] - self.class_means[label])
        centered = np.vstack(centered)
        
        cov_estimator = EmpiricalCovariance(assume_centered=True)
        cov_estimator.fit(centered)
        self.precision = cov_estimator.precision_
    
    def score(self, embeddings):
        min_distances = np.full(len(embeddings), np.inf)
        for label, mean in self.class_means.items():
            diff = embeddings - mean
            distances = np.sum(diff @ self.precision * diff, axis=1)
            min_distances = np.minimum(min_distances, distances)
        return min_distances


# ============================================================================
# Training History Tracker
# ============================================================================

class TrainingHistory:
    """Track and save training/validation metrics over epochs."""
    
    def __init__(self, encoder_name, output_dir):
        self.encoder_name = encoder_name
        self.output_dir = os.path.join(output_dir, encoder_name.lower())
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.train_loss = []
        self.train_acc = []
        self.learning_rates = []
        self.val_auroc = []
        self.val_accuracy = []
        self.val_ece = []
        self.val_nll = []
        
        self.best_epoch = 0
        self.best_auroc = 0.0
        self.start_time = datetime.now()
    
    def log_epoch(self, epoch, train_loss, train_acc, lr, val_metrics):
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.learning_rates.append(lr)
        self.val_auroc.append(val_metrics['auroc'])
        self.val_accuracy.append(val_metrics['accuracy'])
        self.val_ece.append(val_metrics['ece'])
        self.val_nll.append(val_metrics['nll'])
        
        if val_metrics['auroc'] > self.best_auroc:
            self.best_auroc = val_metrics['auroc']
            self.best_epoch = epoch
    
    def save(self):
        history = {
            'encoder': self.encoder_name,
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
        return history
    
    def plot(self):
        epochs = range(1, len(self.train_loss) + 1)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{self.encoder_name} - Linear Probe Training', fontsize=14, fontweight='bold')
        
        # Training Loss
        axes[0, 0].plot(epochs, self.train_loss, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.train_acc, 'g-', linewidth=2, label='Train')
        axes[0, 1].plot(epochs, self.val_accuracy, 'r-', linewidth=2, label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Val AUROC
        axes[0, 2].plot(epochs, self.val_auroc, 'purple', linewidth=2)
        axes[0, 2].axvline(x=self.best_epoch + 1, color='red', linestyle='--', alpha=0.7)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('AUROC')
        axes[0, 2].set_title(f'Validation AUROC (Best: {self.best_auroc:.4f} @ ep{self.best_epoch+1})')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Val ECE
        axes[1, 0].plot(epochs, self.val_ece, 'orange', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('ECE')
        axes[1, 0].set_title('Validation ECE')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Val NLL
        axes[1, 1].plot(epochs, self.val_nll, 'brown', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('NLL')
        axes[1, 1].set_title('Validation NLL')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 2].plot(epochs, self.learning_rates, 'teal', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('LR')
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_linear_probe(model, train_loader, val_loader, device, epochs=50, lr=0.001, 
                       encoder_name='Unknown', output_dir='./results'):
    """Train linear probe with comprehensive logging and saving."""
    
    # Initialize history tracker
    history = TrainingHistory(encoder_name, output_dir)
    
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_auroc = 0.0
    best_state = None
    
    # Print header
    print(f"\n{'='*110}")
    print(f"Training Linear Probe for: {encoder_name}")
    print(f"{'='*110}")
    print(f"{'Epoch':^8} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val AUROC':^10} | {'Val Acc':^10} | {'Val ECE':^10} | {'Val NLL':^10} | {'LR':^12}")
    print(f"{'-'*110}")
    
    for epoch in range(epochs):
        model.head.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
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
        history.log_epoch(epoch, avg_loss, train_acc, current_lr, val_metrics)
        
        # Track best model
        is_best = ""
        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            best_state = {k: v.clone() for k, v in model.head.state_dict().items()}
            is_best = " *BEST*"
            
            # Save best model checkpoint
            best_path = os.path.join(history.output_dir, 'best_linear_probe.pth')
            torch.save({
                'epoch': epoch + 1,
                'head_state_dict': best_state,
                'embed_dim': model.embed_dim,
                'num_classes': 2,
                'encoder_name': encoder_name,
                'val_auroc': best_auroc,
                'val_accuracy': val_metrics['accuracy'],
                'val_ece': val_metrics['ece'],
                'val_nll': val_metrics['nll'],
            }, best_path)
        
        # Print progress
        print(f"{epoch+1:^8} | {avg_loss:^12.4f} | {train_acc:^10.4f} | {val_metrics['auroc']:^10.4f} | {val_metrics['accuracy']:^10.4f} | {val_metrics['ece']:^10.4f} | {val_metrics['nll']:^10.4f} | {current_lr:^12.2e}{is_best}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(history.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'head_state_dict': model.head.state_dict(),
                'embed_dim': model.embed_dim,
                'num_classes': 2,
                'encoder_name': encoder_name,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_loss,
                'val_metrics': val_metrics,
            }, ckpt_path)
    
    print(f"{'-'*110}")
    print(f"Best validation AUROC: {best_auroc:.4f} at epoch {history.best_epoch + 1}")
    print(f"{'='*110}\n")
    
    # Restore best model
    if best_state:
        model.head.load_state_dict(best_state)
    
    # Save final model
    final_path = os.path.join(history.output_dir, 'final_linear_probe.pth')
    torch.save({
        'head_state_dict': model.head.state_dict(),
        'embed_dim': model.embed_dim,
        'num_classes': 2,
        'encoder_name': encoder_name,
        'best_epoch': history.best_epoch + 1,
        'best_auroc': best_auroc,
        'epochs_trained': epochs,
    }, final_path)
    
    # Save history and plot
    history.save()
    history.plot()
    
    logger.info(f"Models saved to {history.output_dir}")
    
    return model, history


def evaluate_model(model, data_loader, device):
    model.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    probs = F.softmax(all_logits, dim=1)
    predictions = probs.argmax(dim=1).numpy()
    labels_np = all_labels.numpy()
    probs_np = probs.numpy()
    
    return {
        'auroc': roc_auc_score(labels_np, probs_np[:, 1]),
        'accuracy': accuracy_score(labels_np, predictions),
        'ece': compute_ece(probs_np, labels_np),
        'nll': compute_nll(all_logits, all_labels)
    }


def extract_features(model, data_loader, device):
    model.eval()
    all_embeddings, all_logits, all_labels = [], [], []
    
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            embeddings = model.get_embeddings(imgs)
            logits = model(imgs)
            all_embeddings.append(embeddings.cpu())
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    return {
        'embeddings': torch.cat(all_embeddings, dim=0).numpy(),
        'logits': torch.cat(all_logits, dim=0),
        'labels': torch.cat(all_labels, dim=0).numpy()
    }


def run_ood_detection(model, id_loader, ood_loaders, device):
    """Run OOD detection for multiple OOD datasets."""
    # Extract ID features
    id_features = extract_features(model, id_loader, device)
    
    # Compute ID scores
    id_energy = compute_energy_score(id_features['logits']).numpy()
    
    # Fit Mahalanobis detector
    mahal = MahalanobisDetector()
    mahal.fit(id_features['embeddings'], id_features['labels'])
    id_mahal = mahal.score(id_features['embeddings'])
    
    results = {}
    for ood_name, ood_loader in ood_loaders.items():
        ood_features = extract_features(model, ood_loader, device)
        
        ood_energy = compute_energy_score(ood_features['logits']).numpy()
        ood_mahal = mahal.score(ood_features['embeddings'])
        
        results[ood_name] = {
            'energy_auroc': compute_ood_auroc(id_energy, ood_energy),
            'energy_fpr95': compute_fpr_at_tpr(id_energy, ood_energy, 0.95),
            'mahal_auroc': compute_ood_auroc(id_mahal, ood_mahal),
            'mahal_fpr95': compute_fpr_at_tpr(id_mahal, ood_mahal, 0.95),
        }
    
    return results


# ============================================================================
# Main Pipeline
# ============================================================================

def run_full_pipeline(encoder_name, encoder, device, data_loaders, epochs=50, output_dir='./results'):
    """Run full evaluation pipeline for one encoder."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running pipeline for: {encoder_name}")
    logger.info(f"{'='*60}")
    
    train_loader, val_loader, id_loader, ood_loaders = data_loaders
    
    # Create encoder output directory
    encoder_output_dir = os.path.join(output_dir, encoder_name.lower())
    os.makedirs(encoder_output_dir, exist_ok=True)
    
    # Create linear probe
    model = LinearProbe(encoder, encoder.embed_dim, num_classes=2).to(device)
    
    # Train linear probe with comprehensive logging
    logger.info("Training linear probe...")
    model, history = train_linear_probe(
        model, train_loader, val_loader, device, 
        epochs=epochs, encoder_name=encoder_name, output_dir=output_dir
    )
    
    # Evaluate on validation set (ID performance)
    logger.info("Evaluating on ID validation set...")
    id_metrics = evaluate_model(model, val_loader, device)
    
    # Run OOD detection
    logger.info("Running OOD detection...")
    ood_metrics = run_ood_detection(model, id_loader, ood_loaders, device)
    
    # Save comprehensive results for this encoder
    encoder_results = {
        'encoder_name': encoder_name,
        'embed_dim': encoder.embed_dim,
        'id_metrics': {k: float(v) for k, v in id_metrics.items()},
        'ood_metrics': {
            ood_name: {k: float(v) for k, v in ood_m.items()}
            for ood_name, ood_m in ood_metrics.items()
        },
        'best_epoch': history.best_epoch + 1,
        'best_val_auroc': history.best_auroc,
    }
    
    results_path = os.path.join(encoder_output_dir, 'full_results.json')
    with open(results_path, 'w') as f:
        json.dump(encoder_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    return {
        'id': id_metrics,
        'ood': ood_metrics,
        'history': history
    }


def main():
    parser = argparse.ArgumentParser(description='Baseline Comparison')
    parser.add_argument('--jepa-checkpoint', type=str, required=True,
                       help='Path to JEPA pretrained checkpoint')
    parser.add_argument('--mae-checkpoint', type=str, default=None,
                       help='Path to MAE pretrained encoder checkpoint')
    parser.add_argument('--supervised-checkpoint', type=str, default=None,
                       help='Path to Supervised pretrained encoder checkpoint')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========================================================================
    # Prepare datasets (same for all encoders)
    # ========================================================================
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ID dataset
    tb_path = os.path.join(args.data_root, 'TB_Chest_Radiography_Database')
    full_dataset = TBClassificationDataset(tb_path, transform=transform)
    
    # Split: 80% train, 20% val (same for all encoders)
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # OOD datasets
    montgomery = MontgomeryDataset(args.data_root, transform=transform)
    shenzhen = ShenzhenDataset(args.data_root, transform=transform)
    
    logger.info(f"ID Train: {n_train}, ID Val: {n_val}")
    logger.info(f"OOD Montgomery: {len(montgomery)}, OOD Shenzhen: {len(shenzhen)}")
    
    # Save dataset split info for reproducibility
    split_info = {
        'n_total': n_total,
        'n_train': n_train,
        'n_val': n_val,
        'train_indices': train_dataset.indices,
        'val_indices': val_dataset.indices,
        'seed': args.seed,
        'data_root': args.data_root,
        'ood_montgomery_size': len(montgomery),
        'ood_shenzhen_size': len(shenzhen)
    }
    split_path = os.path.join(args.output_dir, 'dataset_split.json')
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"Dataset split saved to {split_path}")
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    id_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    ood_loaders = {
        'Montgomery': DataLoader(montgomery, batch_size=args.batch_size, shuffle=False, num_workers=4),
        'Shenzhen': DataLoader(shenzhen, batch_size=args.batch_size, shuffle=False, num_workers=4)
    }
    
    data_loaders = (train_loader, val_loader, id_loader, ood_loaders)
    
    # ========================================================================
    # Run pipeline for each encoder (skip if already completed)
    # ========================================================================
    all_results = {}
    
    def check_experiment_completed(encoder_name, output_dir):
        """Check if experiment already has full_results.json"""
        results_path = os.path.join(output_dir, encoder_name.lower(), 'full_results.json')
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    saved_results = json.load(f)
                # Convert to expected format for comparison table
                return {
                    'id': saved_results['id_metrics'],
                    'ood': saved_results['ood_metrics']
                }
            except Exception as e:
                logger.warning(f"Could not load saved results for {encoder_name}: {e}")
                return None
        return None
    
    # 1. JEPA
    logger.info("\n" + "="*70)
    saved_jepa = check_experiment_completed('JEPA', args.output_dir)
    if saved_jepa:
        logger.info("JEPA experiment already completed - loading saved results")
        all_results['JEPA'] = saved_jepa
    else:
        logger.info("Loading JEPA encoder...")
        jepa_encoder = JEPAEncoder(args.jepa_checkpoint, model_name='vit_small', device=device).to(device)
        all_results['JEPA'] = run_full_pipeline('JEPA', jepa_encoder, device, data_loaders, args.epochs, args.output_dir)
    
    # 2. Supervised (pretrained on TB CXR)
    logger.info("\n" + "="*70)
    saved_supervised = check_experiment_completed('Supervised', args.output_dir)
    if saved_supervised:
        logger.info("Supervised experiment already completed - loading saved results")
        all_results['Supervised'] = saved_supervised
    else:
        logger.info("Loading Supervised encoder...")
        supervised_encoder = SupervisedEncoder(checkpoint_path=args.supervised_checkpoint, device=device).to(device)
        all_results['Supervised'] = run_full_pipeline('Supervised', supervised_encoder, device, data_loaders, args.epochs, args.output_dir)
    
    # 3. MAE (pretrained on TB CXR)
    logger.info("\n" + "="*70)
    saved_mae = check_experiment_completed('MAE', args.output_dir)
    if saved_mae:
        logger.info("MAE experiment already completed - loading saved results")
        all_results['MAE'] = saved_mae
    else:
        logger.info("Loading MAE encoder...")
        mae_encoder = MAEEncoder(checkpoint_path=args.mae_checkpoint, device=device).to(device)
        all_results['MAE'] = run_full_pipeline('MAE', mae_encoder, device, data_loaders, args.epochs, args.output_dir)
    
    # ========================================================================
    # Print comparison table
    # ========================================================================
    print("\n" + "="*120)
    print("COMPARISON TABLE: JEPA vs Supervised vs MAE")
    print("="*120)
    print(f"{'Method':<12} | {'ID AUROC':^10} | {'ID Acc':^10} | {'ID ECE':^10} | {'Mont AUROC':^12} | {'Mont FPR95':^12} | {'Shen AUROC':^12} | {'Shen FPR95':^12}")
    print("-"*120)
    
    for method in ['JEPA', 'Supervised', 'MAE']:
        r = all_results[method]
        id_m = r['id']
        mont = r['ood']['Montgomery']
        shen = r['ood']['Shenzhen']
        
        print(f"{method:<12} | {id_m['auroc']:^10.4f} | {id_m['accuracy']:^10.4f} | {id_m['ece']:^10.4f} | "
              f"{mont['energy_auroc']:^12.4f} | {mont['energy_fpr95']:^12.4f} | "
              f"{shen['energy_auroc']:^12.4f} | {shen['energy_fpr95']:^12.4f}")
    
    print("="*120)
    print("\nNote: OOD metrics shown are for Energy score. Full results saved to JSON.")
    print("="*120)
    
    # ========================================================================
    # Detailed table with both OOD methods
    # ========================================================================
    print("\n" + "="*100)
    print("DETAILED OOD DETECTION RESULTS (Energy & Mahalanobis)")
    print("="*100)
    
    for ood_name in ['Montgomery', 'Shenzhen']:
        print(f"\n{ood_name}:")
        print("-"*80)
        print(f"{'Method':<12} | {'Energy AUROC':^14} | {'Energy FPR95':^14} | {'Mahal AUROC':^14} | {'Mahal FPR95':^14}")
        print("-"*80)
        for method in ['JEPA', 'Supervised', 'MAE']:
            ood = all_results[method]['ood'][ood_name]
            print(f"{method:<12} | {ood['energy_auroc']:^14.4f} | {ood['energy_fpr95']:^14.4f} | "
                  f"{ood['mahal_auroc']:^14.4f} | {ood['mahal_fpr95']:^14.4f}")
    
    print("="*100)
    
    # ========================================================================
    # Save results
    # ========================================================================
    # Convert to JSON-serializable format
    results_json = {
        'experiment_info': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'seed': args.seed,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'id_train_samples': n_train,
            'id_val_samples': n_val,
            'ood_montgomery_samples': len(montgomery),
            'ood_shenzhen_samples': len(shenzhen),
        },
        'methods': {}
    }
    
    for method, r in all_results.items():
        results_json['methods'][method] = {
            'id': {k: float(v) for k, v in r['id'].items()},
            'ood': {
                ood_name: {k: float(v) for k, v in ood_metrics.items()}
                for ood_name, ood_metrics in r['ood'].items()
            },
            'best_epoch': r['history'].best_epoch + 1 if r.get('history') else None,
        }
    
    results_path = os.path.join(args.output_dir, 'baseline_comparison.json')
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"\\nFull comparison results saved to {results_path}")
    
    # Print summary of saved files
    print("\\n" + "="*70)
    print("SAVED FILES SUMMARY")
    print("="*70)
    print(f"Output directory: {args.output_dir}")
    print("-"*70)
    print("Global files:")
    print(f"  - baseline_comparison.json (full comparison results)")
    print(f"  - dataset_split.json (train/val indices)")
    print("-"*70)
    for method in ['JEPA', 'Supervised', 'MAE']:
        method_dir = os.path.join(args.output_dir, method.lower())
        print(f"{method}/ folder:")
        print(f"  - best_linear_probe.pth (best model checkpoint)")
        print(f"  - final_linear_probe.pth (final model)")
        print(f"  - training_history.json (epoch-by-epoch metrics)")
        print(f"  - training_curves.png (visualization)")
        print(f"  - full_results.json (ID + OOD results)")
        print(f"  - checkpoint_epoch_*.pth (periodic checkpoints)")
    print("="*70)
    
    return all_results


if __name__ == '__main__':
    main()

