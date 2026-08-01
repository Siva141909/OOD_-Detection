#!/usr/bin/env python3
"""
OOD Detection for JEPA Embeddings

Domain OOD Detection:
- ID: TB_Chest_Radiography_Database (training data)
- OOD: Montgomery TB, Shenzhen TB (different hospital sources)

OOD Scores:
- Energy Score
- Mahalanobis Distance

Metrics:
- OOD AUROC
- FPR@95%TPR
- OOD ECE
"""

import os
import sys
import argparse
import json
import logging
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.covariance import EmpiricalCovariance
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


class MontgomeryDataset(Dataset):
    """Montgomery TB CXR Dataset."""
    
    SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.image_paths = []
        
        images_folder = os.path.join(root_path, 'Montgomery TB CXR', 'images')
        if os.path.exists(images_folder):
            for f in os.listdir(images_folder):
                if f.endswith(self.SUPPORTED_EXTENSIONS):
                    self.image_paths.append(os.path.join(images_folder, f))
        
        logger.info(f"Montgomery dataset: {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, -1  # No label for OOD


class ShenzhenDataset(Dataset):
    """Shenzhen TB CXR Dataset."""
    
    SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.image_paths = []
        
        # Shenzhen has nested images folder
        images_folder = os.path.join(root_path, 'Shenzhen TB CXR', 'images', 'images')
        if os.path.exists(images_folder):
            for f in os.listdir(images_folder):
                if f.endswith(self.SUPPORTED_EXTENSIONS):
                    self.image_paths.append(os.path.join(images_folder, f))
        
        logger.info(f"Shenzhen dataset: {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, -1  # No label for OOD


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

    def get_embeddings(self, x):
        """Extract pooled embeddings."""
        with torch.no_grad():
            features = self.encoder(x)
            return features.mean(dim=1)


# ============================================================================
# OOD Scores
# ============================================================================

def compute_energy_score(logits, temperature=1.0):
    """
    Compute energy score: -T * log(sum(exp(logits/T)))
    Lower energy = more in-distribution
    We negate so higher score = more OOD
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


class MahalanobisDetector:
    """
    Mahalanobis distance-based OOD detector.
    Fits class-conditional Gaussians on ID embeddings.
    """
    
    def __init__(self):
        self.class_means = {}
        self.precision = None
        self.fitted = False
    
    def fit(self, embeddings, labels):
        """
        Fit class-conditional means and shared covariance.
        
        Args:
            embeddings: [N, D] numpy array
            labels: [N] numpy array
        """
        unique_labels = np.unique(labels)
        
        # Compute class means
        for label in unique_labels:
            mask = labels == label
            self.class_means[label] = embeddings[mask].mean(axis=0)
        
        # Compute centered embeddings for covariance
        centered = []
        for label in unique_labels:
            mask = labels == label
            class_embeddings = embeddings[mask]
            centered.append(class_embeddings - self.class_means[label])
        
        centered = np.vstack(centered)
        
        # Fit covariance with regularization
        cov_estimator = EmpiricalCovariance(assume_centered=True)
        cov_estimator.fit(centered)
        self.precision = cov_estimator.precision_
        
        self.fitted = True
        logger.info(f"Mahalanobis detector fitted on {len(embeddings)} samples, "
                   f"{len(unique_labels)} classes")
    
    def score(self, embeddings):
        """
        Compute Mahalanobis distance to nearest class.
        Higher distance = more OOD.
        
        Args:
            embeddings: [N, D] numpy array
        
        Returns:
            [N] array of Mahalanobis distances
        """
        if not self.fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        min_distances = np.full(len(embeddings), np.inf)
        
        for label, mean in self.class_means.items():
            diff = embeddings - mean
            distances = np.sum(diff @ self.precision * diff, axis=1)
            min_distances = np.minimum(min_distances, distances)
        
        return min_distances


# ============================================================================
# OOD Metrics
# ============================================================================

def compute_ood_auroc(id_scores, ood_scores):
    """
    Compute AUROC for OOD detection.
    Labels: 0 = ID, 1 = OOD
    Higher scores should indicate OOD.
    """
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    return roc_auc_score(labels, scores)


def compute_fpr_at_tpr(id_scores, ood_scores, tpr_threshold=0.95):
    """
    Compute FPR at 95% TPR.
    """
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Find threshold where TPR >= 95%
    idx = np.argmax(tpr >= tpr_threshold)
    return fpr[idx]


def compute_ood_ece(id_probs, ood_probs, n_bins=15):
    """
    Compute ECE treating OOD detection as binary classification.
    Uses max confidence as the OOD score.
    """
    # Combine ID and OOD
    id_conf = np.max(id_probs, axis=1)
    ood_conf = np.max(ood_probs, axis=1)
    
    # Labels: 0 = should be confident (ID), 1 = should be uncertain (OOD)
    # For ECE, we check if high confidence correlates with correct ID detection
    all_conf = np.concatenate([id_conf, ood_conf])
    all_labels = np.concatenate([np.ones(len(id_conf)), np.zeros(len(ood_conf))])  # ID=1, OOD=0
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (all_conf > bin_lower) & (all_conf <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = all_conf[in_bin].mean()
            avg_accuracy = all_labels[in_bin].mean()  # Fraction that are ID
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return ece


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_features(model, data_loader, device):
    """Extract embeddings and logits from a dataset."""
    model.eval()
    
    all_embeddings = []
    all_logits = []
    all_labels = []
    
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


# ============================================================================
# Visualization
# ============================================================================

def plot_ood_distributions(id_scores, ood_scores_dict, score_name, output_path):
    """Plot distribution of OOD scores."""
    n_ood = len(ood_scores_dict)
    fig, axes = plt.subplots(1, n_ood, figsize=(6*n_ood, 5))
    if n_ood == 1:
        axes = [axes]
    
    for ax, (ood_name, ood_scores) in zip(axes, ood_scores_dict.items()):
        ax.hist(id_scores, bins=50, alpha=0.6, label='ID (TB Database)', density=True, color='blue')
        ax.hist(ood_scores, bins=50, alpha=0.6, label=f'OOD ({ood_name})', density=True, color='red')
        ax.set_xlabel(f'{score_name} Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{score_name}: ID vs {ood_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='OOD Detection for JEPA')
    parser.add_argument('--jepa-checkpoint', type=str, required=True)
    parser.add_argument('--probe-checkpoint', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root path to CXR datasets folder')
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
    # 1. Load model
    # ========================================================================
    logger.info("Loading JEPA encoder...")
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
    
    logger.info("Loading linear probe...")
    probe_ckpt = torch.load(args.probe_checkpoint, map_location=device, weights_only=False)
    model = LinearProbe(encoder, embed_dim, num_classes=2).to(device)
    model.head.load_state_dict(probe_ckpt['head_state_dict'])
    model.eval()
    
    # ========================================================================
    # 2. Load datasets
    # ========================================================================
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load split info
    with open(args.split_file, 'r') as f:
        split_info = json.load(f)
    
    # ID dataset (training split)
    tb_path = os.path.join(args.data_root, 'TB_Chest_Radiography_Database')
    full_id_dataset = TBClassificationDataset(tb_path, transform=transform)
    train_indices = split_info['train_indices']
    id_dataset = Subset(full_id_dataset, train_indices)
    
    # OOD datasets
    montgomery_dataset = MontgomeryDataset(args.data_root, transform=transform)
    shenzhen_dataset = ShenzhenDataset(args.data_root, transform=transform)
    
    logger.info(f"ID dataset (training): {len(id_dataset)} samples")
    logger.info(f"OOD Montgomery: {len(montgomery_dataset)} samples")
    logger.info(f"OOD Shenzhen: {len(shenzhen_dataset)} samples")
    
    # Create data loaders
    id_loader = DataLoader(id_dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)
    montgomery_loader = DataLoader(montgomery_dataset, batch_size=args.batch_size,
                                   shuffle=False, num_workers=4, pin_memory=True)
    shenzhen_loader = DataLoader(shenzhen_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True)
    
    # ========================================================================
    # 3. Extract features
    # ========================================================================
    logger.info("Extracting ID features...")
    id_features = extract_features(model, id_loader, device)
    
    logger.info("Extracting Montgomery features...")
    montgomery_features = extract_features(model, montgomery_loader, device)
    
    logger.info("Extracting Shenzhen features...")
    shenzhen_features = extract_features(model, shenzhen_loader, device)
    
    # ========================================================================
    # 4. Compute OOD scores
    # ========================================================================
    
    # --- Energy Score ---
    logger.info("Computing Energy scores...")
    id_energy = compute_energy_score(id_features['logits']).numpy()
    montgomery_energy = compute_energy_score(montgomery_features['logits']).numpy()
    shenzhen_energy = compute_energy_score(shenzhen_features['logits']).numpy()
    
    # --- Mahalanobis Distance ---
    logger.info("Fitting Mahalanobis detector on ID embeddings...")
    mahal_detector = MahalanobisDetector()
    mahal_detector.fit(id_features['embeddings'], id_features['labels'])
    
    id_mahal = mahal_detector.score(id_features['embeddings'])
    montgomery_mahal = mahal_detector.score(montgomery_features['embeddings'])
    shenzhen_mahal = mahal_detector.score(shenzhen_features['embeddings'])
    
    # ========================================================================
    # 5. Get probabilities for ECE
    # ========================================================================
    id_probs = F.softmax(id_features['logits'], dim=1).numpy()
    montgomery_probs = F.softmax(montgomery_features['logits'], dim=1).numpy()
    shenzhen_probs = F.softmax(shenzhen_features['logits'], dim=1).numpy()
    
    # ========================================================================
    # 6. Compute OOD metrics
    # ========================================================================
    results = {}
    
    for ood_name, ood_energy, ood_mahal, ood_probs in [
        ('Montgomery', montgomery_energy, montgomery_mahal, montgomery_probs),
        ('Shenzhen', shenzhen_energy, shenzhen_mahal, shenzhen_probs)
    ]:
        logger.info(f"Computing metrics for {ood_name}...")
        
        results[ood_name] = {
            'energy': {
                'auroc': compute_ood_auroc(id_energy, ood_energy),
                'fpr95': compute_fpr_at_tpr(id_energy, ood_energy, 0.95),
            },
            'mahalanobis': {
                'auroc': compute_ood_auroc(id_mahal, ood_mahal),
                'fpr95': compute_fpr_at_tpr(id_mahal, ood_mahal, 0.95),
            },
            'ece': compute_ood_ece(id_probs, ood_probs),
            'n_samples': len(ood_energy)
        }
    
    # ========================================================================
    # 7. Print results table
    # ========================================================================
    print("\n" + "="*80)
    print("OOD DETECTION RESULTS - JEPA (Domain Shift)")
    print("="*80)
    print(f"ID: TB_Chest_Radiography_Database (n={len(id_dataset)})")
    print("-"*80)
    print(f"{'OOD Dataset':<15} | {'Method':<12} | {'AUROC':^10} | {'FPR@95':^10} | {'ECE':^10}")
    print("-"*80)
    
    for ood_name in ['Montgomery', 'Shenzhen']:
        r = results[ood_name]
        # Energy row
        print(f"{ood_name:<15} | {'Energy':<12} | {r['energy']['auroc']:^10.4f} | {r['energy']['fpr95']:^10.4f} | {r['ece']:^10.4f}")
        # Mahalanobis row
        print(f"{'':<15} | {'Mahalanobis':<12} | {r['mahalanobis']['auroc']:^10.4f} | {r['mahalanobis']['fpr95']:^10.4f} | {'':<10}")
        print("-"*80)
    
    print("="*80)
    print("\nNote: Higher AUROC = better OOD detection, Lower FPR@95 = better")
    print("="*80 + "\n")
    
    # ========================================================================
    # 8. Visualizations
    # ========================================================================
    plot_ood_distributions(
        id_energy, 
        {'Montgomery': montgomery_energy, 'Shenzhen': shenzhen_energy},
        'Energy',
        os.path.join(args.output_dir, 'energy_distribution.png')
    )
    
    plot_ood_distributions(
        id_mahal,
        {'Montgomery': montgomery_mahal, 'Shenzhen': shenzhen_mahal},
        'Mahalanobis',
        os.path.join(args.output_dir, 'mahalanobis_distribution.png')
    )
    
    # ========================================================================
    # 9. Save results
    # ========================================================================
    # Convert numpy to python types for JSON
    results_json = {}
    for ood_name, r in results.items():
        results_json[ood_name] = {
            'energy': {k: float(v) for k, v in r['energy'].items()},
            'mahalanobis': {k: float(v) for k, v in r['mahalanobis'].items()},
            'ece': float(r['ece']),
            'n_samples': r['n_samples']
        }
    
    results_json['id_samples'] = len(id_dataset)
    
    results_path = os.path.join(args.output_dir, 'ood_detection_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    return results


if __name__ == '__main__':
    main()
