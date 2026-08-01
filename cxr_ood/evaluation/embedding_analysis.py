#!/usr/bin/env python3
"""
Subtask 5: Embedding Space Analysis and OOD Detection Evaluation

Reviewer-proof protocol:
- ID data (TB_Database) used for training, validation, calibration
- Montgomery & Shenzhen used ONLY for OOD evaluation (never trained on)

Metrics computed:
1. OOD Detection: AUROC, FPR@95%TPR (ID vs each OOD dataset)
2. OOD Scoring: Energy score, Mahalanobis distance
3. Cross-dataset classification: AUC on OOD with their true labels
4. Embedding analysis: t-SNE, inter-domain distance, intra-class variance
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import cdist
from PIL import Image

# Add IJEPA_Meta to path
IJEPA_PATH = Path(__file__).parent.parent.parent / "IJEPA_Meta"
sys.path.insert(0, str(IJEPA_PATH))
from src.models.vision_transformer import vit_small


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Custom OOD Datasets with Labels (from metadata CSV)
# ============================================================================

class MontgomeryDataset(torch.utils.data.Dataset):
    """Montgomery TB CXR Dataset - OOD test set with labels."""
    SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    
    def __init__(self, root_path, transform=None, return_labels=True):
        self.transform = transform
        self.return_labels = return_labels
        self.samples = []  # (image_path, label)
        
        folder = os.path.join(root_path, 'Montgomery TB CXR', 'images')
        metadata_path = os.path.join(root_path, 'Montgomery TB CXR', 'montgomery_metadata.csv')
        
        # Load metadata for labels
        label_map = {}
        if os.path.exists(metadata_path):
            df = pd.read_csv(metadata_path)
            for _, row in df.iterrows():
                # Label: 0 = Normal, 1 = TB (anything not 'normal')
                label = 0 if row['findings'].strip().lower() == 'normal' else 1
                label_map[row['study_id']] = label
        
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(self.SUPPORTED_EXTENSIONS):
                    img_path = os.path.join(folder, f)
                    # Try to match filename to metadata
                    label = label_map.get(f, -1)  # -1 if not found
                    self.samples.append((img_path, label))
        
        # Count labels
        normal_count = sum(1 for _, l in self.samples if l == 0)
        tb_count = sum(1 for _, l in self.samples if l == 1)
        unknown_count = sum(1 for _, l in self.samples if l == -1)
        logger.info(f"Montgomery: Normal={normal_count}, TB={tb_count}, Unknown={unknown_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class ShenzhenDataset(torch.utils.data.Dataset):
    """Shenzhen TB CXR Dataset - OOD test set with labels."""
    SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    
    def __init__(self, root_path, transform=None, return_labels=True):
        self.transform = transform
        self.return_labels = return_labels
        self.samples = []  # (image_path, label)
        
        folder = os.path.join(root_path, 'Shenzhen TB CXR', 'images', 'images')
        metadata_path = os.path.join(root_path, 'Shenzhen TB CXR', 'shenzhen_metadata.csv')
        
        # Load metadata for labels
        label_map = {}
        if os.path.exists(metadata_path):
            df = pd.read_csv(metadata_path)
            for _, row in df.iterrows():
                # Label: 0 = Normal, 1 = TB
                findings = str(row['findings']).strip().lower()
                label = 0 if findings == 'normal' else 1
                label_map[row['study_id']] = label
        
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(self.SUPPORTED_EXTENSIONS):
                    img_path = os.path.join(folder, f)
                    label = label_map.get(f, -1)
                    self.samples.append((img_path, label))
        
        # Count labels
        normal_count = sum(1 for _, l in self.samples if l == 0)
        tb_count = sum(1 for _, l in self.samples if l == 1)
        unknown_count = sum(1 for _, l in self.samples if l == -1)
        logger.info(f"Shenzhen: Normal={normal_count}, TB={tb_count}, Unknown={unknown_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================================
# Encoder Classes (same as baseline_comparison.py)
# ============================================================================

class JEPAEncoder(nn.Module):
    """JEPA encoder wrapper - uses vit_small with mean pooling."""
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.encoder = vit_small(patch_size=16, drop_path_rate=0.1)
        self.embed_dim = self.encoder.embed_dim
        
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        encoder_state = ckpt.get('target_encoder', ckpt.get('encoder', ckpt))
        
        model_state = self.encoder.state_dict()
        filtered_state = {}
        for k, v in encoder_state.items():
            clean_key = k.replace('module.', '')
            if clean_key in model_state and v.shape == model_state[clean_key].shape:
                filtered_state[clean_key] = v
        
        self.encoder.load_state_dict(filtered_state, strict=False)
        logger.info(f"Loaded JEPA encoder ({len(filtered_state)} params)")
    
    def forward(self, x):
        features = self.encoder(x)
        return features.mean(dim=1)


class MAEEncoder(nn.Module):
    """MAE encoder - uses vit_small() same as JEPA and Supervised."""
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.encoder = vit_small(patch_size=16, drop_path_rate=0.1)
        self.embed_dim = self.encoder.embed_dim
        
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        encoder_state = ckpt.get('encoder', ckpt)
        self.encoder.load_state_dict(encoder_state, strict=True)
        logger.info(f"Loaded MAE encoder ({len(encoder_state)} keys)")
    
    def forward(self, x):
        features = self.encoder(x)
        return features.mean(dim=1)


class SupervisedEncoder(nn.Module):
    """Supervised encoder - uses same vit_small architecture."""
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.encoder = vit_small(patch_size=16, drop_path_rate=0.1)
        self.embed_dim = self.encoder.embed_dim
        
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        encoder_state = ckpt.get('encoder', ckpt)
        
        model_state = self.encoder.state_dict()
        filtered_state = {}
        for k, v in encoder_state.items():
            clean_key = k.replace('module.', '')
            if clean_key in model_state and v.shape == model_state[clean_key].shape:
                filtered_state[clean_key] = v
        
        self.encoder.load_state_dict(filtered_state, strict=False)
        logger.info(f"Loaded Supervised encoder ({len(filtered_state)} params)")
    
    def forward(self, x):
        features = self.encoder(x)
        return features.mean(dim=1)


# ============================================================================
# Linear Classifier for Cross-dataset Evaluation
# ============================================================================

class LinearClassifier(nn.Module):
    """Simple linear classifier on frozen embeddings."""
    def __init__(self, embed_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


# ============================================================================
# Embedding Extraction
# ============================================================================

@torch.no_grad()
def extract_embeddings(encoder, dataloader, device):
    """Extract embeddings for all samples in dataloader."""
    encoder.eval()
    embeddings = []
    labels = []
    
    for images, targets in dataloader:
        images = images.to(device)
        emb = encoder(images)
        embeddings.append(emb.cpu().numpy())
        labels.append(targets.numpy())
    
    return np.vstack(embeddings), np.concatenate(labels)


# ============================================================================
# OOD Detection Scoring Functions
# ============================================================================

def compute_energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Energy-based OOD score: E(x) = -T * log(sum(exp(f(x)/T)))
    Lower energy = more likely in-distribution
    Higher energy = more likely OOD
    """
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    energy = -temperature * torch.logsumexp(logits_tensor / temperature, dim=1)
    return energy.numpy()


def compute_mahalanobis_distance(embeddings: np.ndarray, 
                                  class_means: np.ndarray, 
                                  precision_matrix: np.ndarray) -> np.ndarray:
    """
    Mahalanobis distance-based OOD score.
    Computes minimum Mahalanobis distance to any class centroid.
    Higher distance = more likely OOD
    """
    n_samples = embeddings.shape[0]
    n_classes = class_means.shape[0]
    
    distances = np.zeros((n_samples, n_classes))
    for c in range(n_classes):
        diff = embeddings - class_means[c]
        distances[:, c] = np.sum(diff @ precision_matrix * diff, axis=1)
    
    return np.min(distances, axis=1)


def compute_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    """
    Compute OOD detection metrics.
    Convention: Higher score = more likely OOD
    """
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    
    auroc = roc_auc_score(labels, scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    # First index where TPR >= 95% (not nearest-to-0.95, which can
    # undershoot the 95% guarantee on small sample sizes).
    idx = np.argmax(tpr >= 0.95)
    fpr_at_95_tpr = fpr[idx]
    
    return {
        'auroc': float(auroc),
        'fpr_at_95_tpr': float(fpr_at_95_tpr)
    }


def train_linear_probe(embeddings: np.ndarray, labels: np.ndarray, 
                       embed_dim: int, device: torch.device,
                       epochs: int = 100, lr: float = 0.01) -> LinearClassifier:
    """Train a linear probe on ID embeddings."""
    classifier = LinearClassifier(embed_dim, num_classes=2).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)
    
    classifier.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = classifier(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    
    classifier.eval()
    return classifier


@torch.no_grad()
def get_logits(classifier: LinearClassifier, embeddings: np.ndarray, device: torch.device) -> np.ndarray:
    """Get logits from classifier for given embeddings."""
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    logits = classifier(X)
    return logits.cpu().numpy()


# ============================================================================
# Distance/Variance Metrics
# ============================================================================

def compute_inter_domain_distance(id_embeddings, ood_embeddings):
    """Compute mean distance between ID and OOD embedding distributions."""
    id_centroid = id_embeddings.mean(axis=0)
    ood_centroid = ood_embeddings.mean(axis=0)
    centroid_distance = np.linalg.norm(id_centroid - ood_centroid)
    
    n_samples = min(500, len(id_embeddings), len(ood_embeddings))
    id_sample = id_embeddings[np.random.choice(len(id_embeddings), n_samples, replace=False)]
    ood_sample = ood_embeddings[np.random.choice(len(ood_embeddings), n_samples, replace=False)]
    
    pairwise_dists = cdist(id_sample, ood_sample, metric='euclidean')
    mean_pairwise_distance = pairwise_dists.mean()
    
    return {
        'centroid_distance': float(centroid_distance),
        'mean_pairwise_distance': float(mean_pairwise_distance)
    }


def compute_intra_class_variance(embeddings, labels):
    """Compute variance within each class."""
    unique_labels = np.unique(labels[labels >= 0])
    variances = {}
    
    for label in unique_labels:
        class_embeddings = embeddings[labels == label]
        centroid = class_embeddings.mean(axis=0)
        distances = np.linalg.norm(class_embeddings - centroid, axis=1)
        variance = np.mean(distances ** 2)
        variances[f'class_{int(label)}'] = float(variance)
    
    variances['mean'] = float(np.mean(list(variances.values())))
    return variances


def compute_class_separation(embeddings, labels):
    """Compute separation between class centroids."""
    unique_labels = np.unique(labels[labels >= 0])
    if len(unique_labels) < 2:
        return 0.0
    
    centroids = []
    for label in unique_labels:
        class_embeddings = embeddings[labels == label]
        centroids.append(class_embeddings.mean(axis=0))
    
    return float(np.linalg.norm(centroids[0] - centroids[1]))


# ============================================================================
# Visualization
# ============================================================================

def create_embedding_visualization(
    all_embeddings: dict,
    output_path: str,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42
):
    """Create t-SNE visualization comparing JEPA, MAE, and Supervised embeddings."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = ['JEPA', 'MAE', 'Supervised']
    colors = {
        'ID_Normal': '#2ecc71',
        'ID_TB': '#e74c3c',
        'Montgomery_Normal': '#3498db',
        'Montgomery_TB': '#1a5276',
        'Shenzhen_Normal': '#9b59b6',
        'Shenzhen_TB': '#6c3483'
    }
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        method_data = all_embeddings[method]
        
        combined_embeddings = []
        combined_domains = []
        
        # ID data
        id_emb = method_data['id_embeddings']
        id_labels = method_data['id_labels']
        for i, (emb, label) in enumerate(zip(id_emb, id_labels)):
            combined_embeddings.append(emb)
            domain = 'ID_Normal' if label == 0 else 'ID_TB'
            combined_domains.append(domain)
        
        # Montgomery (with labels)
        mont_emb = method_data['montgomery_embeddings']
        mont_labels = method_data['montgomery_labels']
        for emb, label in zip(mont_emb, mont_labels):
            combined_embeddings.append(emb)
            if label == 0:
                combined_domains.append('Montgomery_Normal')
            elif label == 1:
                combined_domains.append('Montgomery_TB')
            else:
                combined_domains.append('Montgomery_Normal')
        
        # Shenzhen (with labels)
        shen_emb = method_data['shenzhen_embeddings']
        shen_labels = method_data['shenzhen_labels']
        for emb, label in zip(shen_emb, shen_labels):
            combined_embeddings.append(emb)
            if label == 0:
                combined_domains.append('Shenzhen_Normal')
            elif label == 1:
                combined_domains.append('Shenzhen_TB')
            else:
                combined_domains.append('Shenzhen_Normal')
        
        combined_embeddings = np.array(combined_embeddings)
        
        logger.info(f"Running t-SNE for {method}...")
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state,
            init='pca',
            learning_rate='auto'
        )
        embeddings_2d = tsne.fit_transform(combined_embeddings)
        
        for domain in ['ID_Normal', 'ID_TB', 'Montgomery_Normal', 'Montgomery_TB', 'Shenzhen_Normal', 'Shenzhen_TB']:
            mask = np.array([d == domain for d in combined_domains])
            if mask.sum() > 0:
                label_name = domain.replace('_', ' ')
                ax.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=colors[domain],
                    label=label_name,
                    alpha=0.6,
                    s=20,
                    edgecolors='none'
                )
        
        ax.set_title(f'{method}', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=10)
        ax.set_ylabel('t-SNE 2', fontsize=10)
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Embedding Space: ID vs OOD (with class labels)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Visualization saved to {output_path}")


def create_ood_detection_table(all_ood_metrics: dict, output_path: str):
    """Create OOD detection metrics table."""
    
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')
    
    methods = ['JEPA', 'MAE', 'Supervised']
    
    headers = [
        'Method',
        'Mont. Energy\nAUROC ↑',
        'Mont. Energy\nFPR@95 ↓',
        'Mont. Mahal.\nAUROC ↑',
        'Mont. Mahal.\nFPR@95 ↓',
        'Shen. Energy\nAUROC ↑',
        'Shen. Energy\nFPR@95 ↓',
        'Shen. Mahal.\nAUROC ↑',
        'Shen. Mahal.\nFPR@95 ↓'
    ]
    
    rows = []
    for method in methods:
        m = all_ood_metrics[method]
        row = [
            method,
            f"{m['montgomery_energy']['auroc']:.3f}",
            f"{m['montgomery_energy']['fpr_at_95_tpr']:.3f}",
            f"{m['montgomery_mahalanobis']['auroc']:.3f}",
            f"{m['montgomery_mahalanobis']['fpr_at_95_tpr']:.3f}",
            f"{m['shenzhen_energy']['auroc']:.3f}",
            f"{m['shenzhen_energy']['fpr_at_95_tpr']:.3f}",
            f"{m['shenzhen_mahalanobis']['auroc']:.3f}",
            f"{m['shenzhen_mahalanobis']['fpr_at_95_tpr']:.3f}"
        ]
        rows.append(row)
    
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=[0.1] + [0.1125] * 8
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#e74c3c')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i in range(1, len(rows) + 1):
        color = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)
    
    plt.title('OOD Detection Metrics (↑ higher is better for AUROC, ↓ lower is better for FPR)', 
              fontsize=12, fontweight='bold', pad=20)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"OOD detection table saved to {output_path}")


def create_cross_dataset_table(all_cross_metrics: dict, output_path: str):
    """Create cross-dataset classification table."""
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    
    methods = ['JEPA', 'MAE', 'Supervised']
    
    headers = [
        'Method',
        'ID Val\nAUC ↑',
        'ID Val\nAcc ↑',
        'Montgomery\nAUC ↑',
        'Montgomery\nAcc ↑',
        'Shenzhen\nAUC ↑',
        'Shenzhen\nAcc ↑'
    ]
    
    rows = []
    for method in methods:
        m = all_cross_metrics[method]
        row = [
            method,
            f"{m['id_auc']:.3f}",
            f"{m['id_acc']:.3f}",
            f"{m['montgomery_auc']:.3f}",
            f"{m['montgomery_acc']:.3f}",
            f"{m['shenzhen_auc']:.3f}",
            f"{m['shenzhen_acc']:.3f}"
        ]
        rows.append(row)
    
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=[0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#27ae60')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i in range(1, len(rows) + 1):
        color = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Cross-Dataset Classification (Linear Probe trained on ID only, evaluated on OOD)', 
              fontsize=12, fontweight='bold', pad=20)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Cross-dataset table saved to {output_path}")


def print_all_results(all_stats: dict, all_ood_metrics: dict, all_cross_metrics: dict):
    """Print comprehensive results to console."""
    
    methods = ['JEPA', 'MAE', 'Supervised']
    
    print("\n" + "=" * 120)
    print("COMPREHENSIVE OOD DETECTION & EMBEDDING ANALYSIS RESULTS")
    print("=" * 120)
    
    # 1. OOD Detection Metrics
    print("\n" + "-" * 120)
    print("1. OOD DETECTION METRICS (ID vs OOD classification)")
    print("-" * 120)
    print(f"{'Method':<12} | {'Montgomery Energy':<22} | {'Montgomery Mahalanobis':<22} | {'Shenzhen Energy':<22} | {'Shenzhen Mahalanobis':<22}")
    print(f"{'':12} | {'AUROC':>10} {'FPR@95':>10} | {'AUROC':>10} {'FPR@95':>10} | {'AUROC':>10} {'FPR@95':>10} | {'AUROC':>10} {'FPR@95':>10}")
    print("-" * 120)
    for method in methods:
        m = all_ood_metrics[method]
        print(f"{method:<12} | {m['montgomery_energy']['auroc']:>10.4f} {m['montgomery_energy']['fpr_at_95_tpr']:>10.4f} | "
              f"{m['montgomery_mahalanobis']['auroc']:>10.4f} {m['montgomery_mahalanobis']['fpr_at_95_tpr']:>10.4f} | "
              f"{m['shenzhen_energy']['auroc']:>10.4f} {m['shenzhen_energy']['fpr_at_95_tpr']:>10.4f} | "
              f"{m['shenzhen_mahalanobis']['auroc']:>10.4f} {m['shenzhen_mahalanobis']['fpr_at_95_tpr']:>10.4f}")
    
    # 2. Cross-Dataset Classification
    print("\n" + "-" * 120)
    print("2. CROSS-DATASET CLASSIFICATION (Linear probe trained on ID, tested on OOD)")
    print("-" * 120)
    print(f"{'Method':<12} | {'ID Validation':<20} | {'Montgomery (OOD)':<20} | {'Shenzhen (OOD)':<20}")
    print(f"{'':12} | {'AUC':>10} {'Acc':>8} | {'AUC':>10} {'Acc':>8} | {'AUC':>10} {'Acc':>8}")
    print("-" * 120)
    for method in methods:
        m = all_cross_metrics[method]
        print(f"{method:<12} | {m['id_auc']:>10.4f} {m['id_acc']:>8.4f} | "
              f"{m['montgomery_auc']:>10.4f} {m['montgomery_acc']:>8.4f} | "
              f"{m['shenzhen_auc']:>10.4f} {m['shenzhen_acc']:>8.4f}")
    
    # 3. Embedding Space Analysis
    print("\n" + "-" * 120)
    print("3. EMBEDDING SPACE METRICS")
    print("-" * 120)
    print(f"{'Method':<12} | {'ID Class Sep':>12} | {'ID Variance':>12} | {'Mont. Dist':>12} | {'Shen. Dist':>12}")
    print("-" * 120)
    for method in methods:
        s = all_stats[method]
        print(f"{method:<12} | {s['id_class_separation']:>12.4f} | {s['id_intra_class_variance']['mean']:>12.4f} | "
              f"{s['montgomery_inter_domain']['centroid_distance']:>12.4f} | {s['shenzhen_inter_domain']['centroid_distance']:>12.4f}")
    
    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY: Best Method for Each Metric")
    print("=" * 120)
    
    # Best OOD detection (higher AUROC is better)
    best_mont_energy = max(methods, key=lambda m: all_ood_metrics[m]['montgomery_energy']['auroc'])
    best_shen_energy = max(methods, key=lambda m: all_ood_metrics[m]['shenzhen_energy']['auroc'])
    best_mont_mahal = max(methods, key=lambda m: all_ood_metrics[m]['montgomery_mahalanobis']['auroc'])
    best_shen_mahal = max(methods, key=lambda m: all_ood_metrics[m]['shenzhen_mahalanobis']['auroc'])
    
    print(f"  ✓ Best Montgomery OOD (Energy): {best_mont_energy} (AUROC={all_ood_metrics[best_mont_energy]['montgomery_energy']['auroc']:.4f})")
    print(f"  ✓ Best Shenzhen OOD (Energy): {best_shen_energy} (AUROC={all_ood_metrics[best_shen_energy]['shenzhen_energy']['auroc']:.4f})")
    print(f"  ✓ Best Montgomery OOD (Mahalanobis): {best_mont_mahal} (AUROC={all_ood_metrics[best_mont_mahal]['montgomery_mahalanobis']['auroc']:.4f})")
    print(f"  ✓ Best Shenzhen OOD (Mahalanobis): {best_shen_mahal} (AUROC={all_ood_metrics[best_shen_mahal]['shenzhen_mahalanobis']['auroc']:.4f})")
    
    # Best cross-dataset (higher AUC is better)
    best_mont_class = max(methods, key=lambda m: all_cross_metrics[m]['montgomery_auc'])
    best_shen_class = max(methods, key=lambda m: all_cross_metrics[m]['shenzhen_auc'])
    
    print(f"  ✓ Best Montgomery Classification: {best_mont_class} (AUC={all_cross_metrics[best_mont_class]['montgomery_auc']:.4f})")
    print(f"  ✓ Best Shenzhen Classification: {best_shen_class} (AUC={all_cross_metrics[best_shen_class]['shenzhen_auc']:.4f})")
    
    print("=" * 120)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='OOD Detection & Embedding Analysis')
    parser.add_argument('--jepa-checkpoint', type=str, required=True)
    parser.add_argument('--mae-checkpoint', type=str, required=True)
    parser.add_argument('--supervised-checkpoint', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--n-iter', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    id_dataset = ImageFolder(os.path.join(args.data_root, 'TB_Chest_Radiography_Database'), transform=transform)
    montgomery_dataset = MontgomeryDataset(args.data_root, transform=transform)
    shenzhen_dataset = ShenzhenDataset(args.data_root, transform=transform)
    
    logger.info(f"ID dataset: {len(id_dataset)} samples")
    logger.info(f"Montgomery (OOD): {len(montgomery_dataset)} samples")
    logger.info(f"Shenzhen (OOD): {len(shenzhen_dataset)} samples")
    
    # Load validation split
    baseline_dir = os.path.dirname(args.output_dir)
    split_path = os.path.join(baseline_dir, 'baseline_comparison', 'dataset_split.json')
    if not os.path.exists(split_path):
        split_path = os.path.join(args.output_dir, '..', 'baseline_comparison', 'dataset_split.json')
    
    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            split = json.load(f)
        val_indices = split['val_indices']
        id_subset = Subset(id_dataset, val_indices)
        logger.info(f"Using validation split: {len(id_subset)} samples")
    else:
        id_subset = Subset(id_dataset, list(range(min(840, len(id_dataset)))))
        logger.info(f"Using first {len(id_subset)} samples as ID subset")
    
    # Create dataloaders
    id_loader = DataLoader(id_subset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    montgomery_loader = DataLoader(montgomery_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    shenzhen_loader = DataLoader(shenzhen_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load encoders
    encoders = {
        'JEPA': JEPAEncoder(args.jepa_checkpoint).to(device),
        'MAE': MAEEncoder(args.mae_checkpoint).to(device),
        'Supervised': SupervisedEncoder(args.supervised_checkpoint).to(device)
    }
    
    # Results storage
    all_embeddings = {}
    all_stats = {}
    all_ood_metrics = {}
    all_cross_metrics = {}
    
    for method, encoder in encoders.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {method}...")
        logger.info(f"{'='*60}")
        
        embed_dim = encoder.embed_dim
        
        # Extract embeddings
        logger.info("Extracting embeddings...")
        id_emb, id_labels = extract_embeddings(encoder, id_loader, device)
        mont_emb, mont_labels = extract_embeddings(encoder, montgomery_loader, device)
        shen_emb, shen_labels = extract_embeddings(encoder, shenzhen_loader, device)
        
        logger.info(f"  ID: {id_emb.shape}, Montgomery: {mont_emb.shape}, Shenzhen: {shen_emb.shape}")
        
        all_embeddings[method] = {
            'id_embeddings': id_emb,
            'id_labels': id_labels,
            'montgomery_embeddings': mont_emb,
            'montgomery_labels': mont_labels,
            'shenzhen_embeddings': shen_emb,
            'shenzhen_labels': shen_labels
        }
        
        # ===== 1. Train linear probe on ID =====
        logger.info("Training linear probe on ID embeddings...")
        classifier = train_linear_probe(id_emb, id_labels, embed_dim, device)
        
        # Get logits
        id_logits = get_logits(classifier, id_emb, device)
        mont_logits = get_logits(classifier, mont_emb, device)
        shen_logits = get_logits(classifier, shen_emb, device)
        
        # ===== 2. OOD Detection with Energy Score =====
        logger.info("Computing Energy-based OOD scores...")
        id_energy = compute_energy_score(id_logits)
        mont_energy = compute_energy_score(mont_logits)
        shen_energy = compute_energy_score(shen_logits)
        
        mont_energy_metrics = compute_ood_metrics(id_energy, mont_energy)
        shen_energy_metrics = compute_ood_metrics(id_energy, shen_energy)
        
        # ===== 3. OOD Detection with Mahalanobis Distance =====
        logger.info("Computing Mahalanobis-based OOD scores...")
        
        # Compute class means and shared covariance from ID data only
        class_means = []
        for c in [0, 1]:
            class_emb = id_emb[id_labels == c]
            class_means.append(class_emb.mean(axis=0))
        class_means = np.array(class_means)
        
        # Shared covariance (pooled)
        cov = EmpiricalCovariance().fit(id_emb)
        precision = cov.precision_
        
        id_mahal = compute_mahalanobis_distance(id_emb, class_means, precision)
        mont_mahal = compute_mahalanobis_distance(mont_emb, class_means, precision)
        shen_mahal = compute_mahalanobis_distance(shen_emb, class_means, precision)
        
        mont_mahal_metrics = compute_ood_metrics(id_mahal, mont_mahal)
        shen_mahal_metrics = compute_ood_metrics(id_mahal, shen_mahal)
        
        all_ood_metrics[method] = {
            'montgomery_energy': mont_energy_metrics,
            'shenzhen_energy': shen_energy_metrics,
            'montgomery_mahalanobis': mont_mahal_metrics,
            'shenzhen_mahalanobis': shen_mahal_metrics
        }
        
        # ===== 4. Cross-Dataset Classification =====
        logger.info("Evaluating cross-dataset classification...")
        
        # ID validation metrics
        id_probs = torch.softmax(torch.tensor(id_logits), dim=1)[:, 1].numpy()
        id_preds = np.argmax(id_logits, axis=1)
        id_auc = roc_auc_score(id_labels, id_probs)
        id_acc = accuracy_score(id_labels, id_preds)
        
        # Montgomery (filter out unknown labels)
        mont_valid = mont_labels >= 0
        if mont_valid.sum() > 0:
            mont_probs = torch.softmax(torch.tensor(mont_logits[mont_valid]), dim=1)[:, 1].numpy()
            mont_preds = np.argmax(mont_logits[mont_valid], axis=1)
            mont_auc = roc_auc_score(mont_labels[mont_valid], mont_probs)
            mont_acc = accuracy_score(mont_labels[mont_valid], mont_preds)
        else:
            mont_auc, mont_acc = 0.5, 0.5
        
        # Shenzhen (filter out unknown labels)
        shen_valid = shen_labels >= 0
        if shen_valid.sum() > 0:
            shen_probs = torch.softmax(torch.tensor(shen_logits[shen_valid]), dim=1)[:, 1].numpy()
            shen_preds = np.argmax(shen_logits[shen_valid], axis=1)
            shen_auc = roc_auc_score(shen_labels[shen_valid], shen_probs)
            shen_acc = accuracy_score(shen_labels[shen_valid], shen_preds)
        else:
            shen_auc, shen_acc = 0.5, 0.5
        
        all_cross_metrics[method] = {
            'id_auc': float(id_auc),
            'id_acc': float(id_acc),
            'montgomery_auc': float(mont_auc),
            'montgomery_acc': float(mont_acc),
            'shenzhen_auc': float(shen_auc),
            'shenzhen_acc': float(shen_acc)
        }
        
        # ===== 5. Embedding Statistics =====
        all_stats[method] = {
            'id_class_separation': compute_class_separation(id_emb, id_labels),
            'id_intra_class_variance': compute_intra_class_variance(id_emb, id_labels),
            'montgomery_inter_domain': compute_inter_domain_distance(id_emb, mont_emb),
            'shenzhen_inter_domain': compute_inter_domain_distance(id_emb, shen_emb)
        }
        
        logger.info(f"  ID AUC: {id_auc:.4f}, Mont AUC: {mont_auc:.4f}, Shen AUC: {shen_auc:.4f}")
        logger.info(f"  Mont OOD AUROC (Energy): {mont_energy_metrics['auroc']:.4f}")
        logger.info(f"  Shen OOD AUROC (Energy): {shen_energy_metrics['auroc']:.4f}")
    
    # Print comprehensive results
    print_all_results(all_stats, all_ood_metrics, all_cross_metrics)
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    viz_path = os.path.join(args.output_dir, 'embedding_visualization.png')
    create_embedding_visualization(all_embeddings, viz_path, 
                                   perplexity=args.perplexity, n_iter=args.n_iter, random_state=args.seed)
    
    ood_table_path = os.path.join(args.output_dir, 'ood_detection_metrics.png')
    create_ood_detection_table(all_ood_metrics, ood_table_path)
    
    cross_table_path = os.path.join(args.output_dir, 'cross_dataset_classification.png')
    create_cross_dataset_table(all_cross_metrics, cross_table_path)
    
    # Save all results to JSON
    results = {
        'embedding_statistics': all_stats,
        'ood_detection_metrics': all_ood_metrics,
        'cross_dataset_metrics': all_cross_metrics
    }
    
    results_path = os.path.join(args.output_dir, 'full_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print output files
    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print(f"  - {viz_path}")
    print(f"  - {ood_table_path}")
    print(f"  - {cross_table_path}")
    print(f"  - {results_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
