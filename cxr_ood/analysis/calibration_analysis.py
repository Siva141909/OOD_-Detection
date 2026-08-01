#!/usr/bin/env python3
"""
Calibration Analysis for JEPA Linear Probe

Analyzes calibration before and after temperature scaling:
1. Load trained linear probe
2. Compute ECE and reliability diagram (before calibration)
3. Apply temperature scaling on validation set
4. Recompute ECE and NLL (after calibration)
5. Output comparison table and reliability plot
"""

import os
import sys
import argparse
import json
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add IJEPA_Meta to path
IJEPA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "IJEPA_Meta")
sys.path.insert(0, IJEPA_PATH)
from src.models.vision_transformer import vit_small, vit_base, vit_large, vit_huge

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset
# ============================================================================

class TBClassificationDataset(Dataset):
    """TB Chest Radiography Database with labels."""
    
    SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        normal_folder = os.path.join(root_path, 'Normal')
        if os.path.exists(normal_folder):
            for f in os.listdir(normal_folder):
                if f.endswith(self.SUPPORTED_EXTENSIONS):
                    self.image_paths.append(os.path.join(normal_folder, f))
                    self.labels.append(0)
        
        tb_folder = os.path.join(root_path, 'Tuberculosis')
        if os.path.exists(tb_folder):
            for f in os.listdir(tb_folder):
                if f.endswith(self.SUPPORTED_EXTENSIONS):
                    self.image_paths.append(os.path.join(tb_folder, f))
                    self.labels.append(1)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ============================================================================
# Linear Probe Model
# ============================================================================

class LinearProbe(nn.Module):
    """Frozen encoder + linear head."""
    
    def __init__(self, encoder, embed_dim, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.head = nn.Linear(embed_dim, num_classes)
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
    
    def forward(self, x):
        # This backbone has no CLS token (cls_token=False in IJEPA's ViT);
        # mean-pool over patch tokens for a global representation.
        with torch.no_grad():
            features = self.encoder(x)
            pooled = features.mean(dim=1)
        return self.head(pooled)


# ============================================================================
# Temperature Scaling
# ============================================================================

class TemperatureScaling(nn.Module):
    """
    Temperature scaling for calibration.
    Learns a single temperature parameter to scale logits.
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        return logits / self.temperature
    
    def fit(self, logits, labels, lr=0.01, max_iter=50):
        """
        Optimize temperature on validation set using NLL.
        """
        self.temperature.data = torch.ones(1, device=logits.device) * 1.5
        
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return self.temperature.item()


# ============================================================================
# Calibration Metrics
# ============================================================================

def compute_ece(probs, labels, n_bins=15):
    """Compute Expected Calibration Error."""
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return ece


def compute_nll(logits, labels):
    """Compute Negative Log-Likelihood."""
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    return F.cross_entropy(logits, labels, reduction='mean').item()


def compute_brier_score(probs, labels):
    """Compute Brier Score (MSE of probabilities)."""
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # One-hot encode labels
    n_classes = probs.shape[1]
    labels_onehot = np.eye(n_classes)[labels]
    
    return np.mean(np.sum((probs - labels_onehot) ** 2, axis=1))


def get_reliability_data(probs, labels, n_bins=10):
    """Get data for reliability diagram."""
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.sum() > 0:
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_confidences.append(confidences[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append(bin_centers[len(bin_accuracies)])
            bin_counts.append(0)
    
    return {
        'bin_centers': bin_centers,
        'bin_accuracies': np.array(bin_accuracies),
        'bin_confidences': np.array(bin_confidences),
        'bin_counts': np.array(bin_counts)
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_reliability_diagram(before_data, after_data, output_path, ece_before, ece_after):
    """Plot reliability diagram before and after temperature scaling."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before calibration
    ax = axes[0]
    n_bins = len(before_data['bin_centers'])
    width = 0.8 / n_bins
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    
    # Bar plot
    ax.bar(before_data['bin_confidences'], before_data['bin_accuracies'], 
           width=0.08, alpha=0.7, color='steelblue', edgecolor='black', label='Accuracy')
    
    # Gap visualization
    for i, (conf, acc) in enumerate(zip(before_data['bin_confidences'], before_data['bin_accuracies'])):
        if before_data['bin_counts'][i] > 0:
            gap_color = 'red' if conf > acc else 'green'
            ax.plot([conf, conf], [acc, conf], color=gap_color, linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Before Temperature Scaling\nECE = {ece_before:.4f}', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # After calibration
    ax = axes[1]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    ax.bar(after_data['bin_confidences'], after_data['bin_accuracies'], 
           width=0.08, alpha=0.7, color='forestgreen', edgecolor='black', label='Accuracy')
    
    for i, (conf, acc) in enumerate(zip(after_data['bin_confidences'], after_data['bin_accuracies'])):
        if after_data['bin_counts'][i] > 0:
            gap_color = 'red' if conf > acc else 'green'
            ax.plot([conf, conf], [acc, conf], color=gap_color, linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'After Temperature Scaling\nECE = {ece_after:.4f}', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Reliability diagram saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Calibration Analysis')
    parser.add_argument('--jepa-checkpoint', type=str, required=True,
                       help='Path to JEPA encoder checkpoint')
    parser.add_argument('--probe-checkpoint', type=str, required=True,
                       help='Path to trained linear probe checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to TB dataset')
    parser.add_argument('--split-file', type=str, required=True,
                       help='Path to dataset_split.json from training')
    parser.add_argument('--model-name', type=str, default='vit_small')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========================================================================
    # 1. Load JEPA encoder
    # ========================================================================
    logger.info(f"Loading JEPA encoder from {args.jepa_checkpoint}")
    checkpoint = torch.load(args.jepa_checkpoint, map_location=device, weights_only=False)
    
    model_fn = {'vit_small': vit_small, 'vit_base': vit_base, 
                'vit_large': vit_large, 'vit_huge': vit_huge}[args.model_name]
    encoder = model_fn()
    
    if 'target_encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['target_encoder'])
    else:
        encoder.load_state_dict(checkpoint['encoder'])
    
    encoder = encoder.to(device)
    encoder.eval()
    embed_dim = encoder.embed_dim
    
    # ========================================================================
    # 2. Load trained linear probe
    # ========================================================================
    logger.info(f"Loading linear probe from {args.probe_checkpoint}")
    probe_ckpt = torch.load(args.probe_checkpoint, map_location=device, weights_only=False)
    
    model = LinearProbe(encoder, embed_dim, num_classes=2).to(device)
    model.head.load_state_dict(probe_ckpt['head_state_dict'])
    model.eval()
    
    # ========================================================================
    # 3. Load dataset with same split as training
    # ========================================================================
    with open(args.split_file, 'r') as f:
        split_info = json.load(f)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = TBClassificationDataset(args.data_path, transform=transform)
    
    # Use the exact same validation indices
    val_indices = split_info['val_indices']
    val_dataset = Subset(full_dataset, val_indices)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    logger.info(f"Validation set size: {len(val_dataset)}")
    
    # ========================================================================
    # 4. Collect predictions on validation set
    # ========================================================================
    logger.info("Collecting predictions...")
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # ========================================================================
    # 5. Compute metrics BEFORE temperature scaling
    # ========================================================================
    logger.info("Computing metrics before temperature scaling...")
    
    probs_before = F.softmax(all_logits, dim=1)
    
    ece_before = compute_ece(probs_before, all_labels)
    nll_before = compute_nll(all_logits, all_labels)
    brier_before = compute_brier_score(probs_before, all_labels)
    accuracy = accuracy_score(all_labels.numpy(), probs_before.argmax(dim=1).numpy())
    auroc = roc_auc_score(all_labels.numpy(), probs_before[:, 1].numpy())
    
    reliability_before = get_reliability_data(probs_before, all_labels)
    
    # ========================================================================
    # 6. Apply temperature scaling
    # ========================================================================
    logger.info("Fitting temperature scaling...")
    
    temp_scaler = TemperatureScaling().to(device)
    logits_device = all_logits.to(device)
    labels_device = all_labels.to(device)
    
    optimal_temp = temp_scaler.fit(logits_device, labels_device)
    logger.info(f"Optimal temperature: {optimal_temp:.4f}")
    
    # ========================================================================
    # 7. Compute metrics AFTER temperature scaling
    # ========================================================================
    logger.info("Computing metrics after temperature scaling...")
    
    with torch.no_grad():
        scaled_logits = temp_scaler(logits_device).cpu()
    probs_after = F.softmax(scaled_logits, dim=1)
    
    ece_after = compute_ece(probs_after, all_labels)
    nll_after = compute_nll(scaled_logits, all_labels)
    brier_after = compute_brier_score(probs_after, all_labels)
    
    reliability_after = get_reliability_data(probs_after, all_labels)
    
    # ========================================================================
    # 8. Print results table
    # ========================================================================
    print("\n" + "="*70)
    print("CALIBRATION ANALYSIS - In-Domain (TB_Chest_Radiography_Database)")
    print("="*70)
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Accuracy: {accuracy:.4f} | AUROC: {auroc:.4f}")
    print(f"Optimal Temperature: {optimal_temp:.4f}")
    print("-"*70)
    print(f"{'Metric':<25} {'Before T-Scaling':<20} {'After T-Scaling':<20}")
    print("-"*70)
    print(f"{'ECE':<25} {ece_before:<20.4f} {ece_after:<20.4f}")
    print(f"{'NLL':<25} {nll_before:<20.4f} {nll_after:<20.4f}")
    print(f"{'Brier Score':<25} {brier_before:<20.4f} {brier_after:<20.4f}")
    print("-"*70)
    
    # Improvement percentages
    ece_improvement = (ece_before - ece_after) / ece_before * 100 if ece_before > 0 else 0
    nll_improvement = (nll_before - nll_after) / nll_before * 100 if nll_before > 0 else 0
    
    print(f"{'ECE Improvement':<25} {ece_improvement:+.1f}%")
    print(f"{'NLL Improvement':<25} {nll_improvement:+.1f}%")
    print("="*70 + "\n")
    
    # ========================================================================
    # 9. Plot reliability diagram
    # ========================================================================
    plot_path = os.path.join(args.output_dir, 'reliability_diagram.png')
    plot_reliability_diagram(reliability_before, reliability_after, plot_path, 
                            ece_before, ece_after)
    
    # ========================================================================
    # 10. Save results
    # ========================================================================
    results = {
        'validation_samples': len(val_dataset),
        'accuracy': float(accuracy),
        'auroc': float(auroc),
        'optimal_temperature': float(optimal_temp),
        'before_calibration': {
            'ece': float(ece_before),
            'nll': float(nll_before),
            'brier_score': float(brier_before)
        },
        'after_calibration': {
            'ece': float(ece_after),
            'nll': float(nll_after),
            'brier_score': float(brier_after)
        },
        'improvement': {
            'ece_percent': float(ece_improvement),
            'nll_percent': float(nll_improvement)
        }
    }
    
    results_path = os.path.join(args.output_dir, 'calibration_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Save temperature for later use
    temp_path = os.path.join(args.output_dir, 'temperature_scaling.pth')
    torch.save({
        'temperature': optimal_temp,
        'head_state_dict': model.head.state_dict()
    }, temp_path)
    logger.info(f"Temperature scaling saved to {temp_path}")
    
    return results


if __name__ == '__main__':
    main()
