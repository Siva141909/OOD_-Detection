# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for Chest X-Ray datasets for Medical JEPA pretraining

import os
import glob
from logging import getLogger
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logger = getLogger()


class CXRDataset(Dataset):
    """
    Unified Chest X-Ray Dataset for JEPA pretraining.
    
    Combines multiple CXR datasets:
    - Kermany's CXR (Pneumonia/Normal)
    - Montgomery TB CXR
    - Shenzhen TB CXR
    - TB Chest Radiography Database
    
    For JEPA pretraining, labels are not needed - we only use images.
    """
    
    SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    
    def __init__(
        self,
        root_path,
        transform=None,
        datasets_to_use=None,
        max_images=None
    ):
        """
        Args:
            root_path: Path to the CXR datasets directory
            transform: Image transformations to apply
            datasets_to_use: List of dataset names to include. If None, use all.
                Options: ['kermany', 'montgomery', 'shenzhen', 'tb_database']
            max_images: Maximum number of images to use (for debugging/testing)
        """
        self.root_path = root_path
        self.transform = transform
        self.datasets_to_use = datasets_to_use or ['kermany', 'montgomery', 'shenzhen', 'tb_database']
        
        self.image_paths = []
        self.dataset_sources = []  # Track which dataset each image came from
        
        # Collect images from each dataset
        self._collect_images()
        
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
            self.dataset_sources = self.dataset_sources[:max_images]
        
        logger.info(f'CXR Dataset initialized with {len(self.image_paths)} images')
        self._log_dataset_stats()
    
    def _collect_images(self):
        """Collect image paths from all specified datasets."""
        
        # Kermany's CXR dataset
        if 'kermany' in self.datasets_to_use:
            # Try both apostrophe variants (straight ' and curly ' U+2019)
            kermany_names = ["Kermany's CXR", "Kermany\u2019s CXR"]
            kermany_found = False
            for name in kermany_names:
                kermany_path = os.path.join(self.root_path, name)
                if os.path.exists(kermany_path):
                    kermany_found = True
                    for split in ['train', 'test']:
                        for label in ['NORMAL', 'PNEUMONIA']:
                            folder = os.path.join(kermany_path, split, label)
                            if os.path.exists(folder):
                                self._add_images_from_folder(folder, 'kermany')
                    break
            if not kermany_found:
                logger.warning(f"Kermany dataset not found (tried both apostrophe variants)")
        
        # Montgomery TB CXR dataset
        if 'montgomery' in self.datasets_to_use:
            montgomery_path = os.path.join(self.root_path, "Montgomery TB CXR", "images")
            if os.path.exists(montgomery_path):
                self._add_images_from_folder(montgomery_path, 'montgomery')
            else:
                logger.warning(f"Montgomery dataset not found at {montgomery_path}")
        
        # Shenzhen TB CXR dataset
        if 'shenzhen' in self.datasets_to_use:
            shenzhen_path = os.path.join(self.root_path, "Shenzhen TB CXR", "images", "images")
            if os.path.exists(shenzhen_path):
                self._add_images_from_folder(shenzhen_path, 'shenzhen')
            else:
                logger.warning(f"Shenzhen dataset not found at {shenzhen_path}")
        
        # TB Chest Radiography Database
        if 'tb_database' in self.datasets_to_use:
            tb_path = os.path.join(self.root_path, "TB_Chest_Radiography_Database")
            if os.path.exists(tb_path):
                for label in ['Normal', 'Tuberculosis']:
                    folder = os.path.join(tb_path, label)
                    if os.path.exists(folder):
                        self._add_images_from_folder(folder, 'tb_database')
            else:
                logger.warning(f"TB Database not found at {tb_path}")
    
    def _add_images_from_folder(self, folder, source_name):
        """Add all valid images from a folder."""
        for ext in self.SUPPORTED_EXTENSIONS:
            pattern = os.path.join(folder, f'*{ext}')
            files = glob.glob(pattern)
            self.image_paths.extend(files)
            self.dataset_sources.extend([source_name] * len(files))
    
    def _log_dataset_stats(self):
        """Log statistics about the collected dataset."""
        from collections import Counter
        source_counts = Counter(self.dataset_sources)
        logger.info("Dataset composition:")
        for source, count in source_counts.items():
            logger.info(f"  {source}: {count} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # Load image and convert to RGB (CXR images are often grayscale)
            img = Image.open(img_path)
            
            # Handle different image modes
            if img.mode == 'L':  # Grayscale
                img = img.convert('RGB')
            elif img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            if self.transform is not None:
                img = self.transform(img)
            
            return img
            
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            # Return a random other image if this one fails
            return self.__getitem__((idx + 1) % len(self))


def make_cxr_dataset(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=4,
    world_size=1,
    rank=0,
    root_path=None,
    datasets_to_use=None,
    max_images=None,
    drop_last=True,
    shuffle=True
):
    """
    Create CXR dataset and dataloader for JEPA pretraining.
    
    Args:
        transform: Image transformations
        batch_size: Batch size per GPU
        collator: Custom collator (e.g., MaskCollator for JEPA)
        pin_mem: Pin memory for faster GPU transfer
        num_workers: Number of data loading workers
        world_size: Number of distributed processes
        rank: Rank of current process
        root_path: Path to CXR datasets directory
        datasets_to_use: List of datasets to include
        max_images: Maximum images to use
        drop_last: Drop last incomplete batch
        shuffle: Shuffle the dataset
    
    Returns:
        dataset, dataloader, sampler
    """
    dataset = CXRDataset(
        root_path=root_path,
        transform=transform,
        datasets_to_use=datasets_to_use,
        max_images=max_images
    )
    
    logger.info(f'CXR dataset created with {len(dataset)} images')
    
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
    else:
        sampler = None
    
    data_loader = DataLoader(
        dataset,
        collate_fn=collator,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        shuffle=(shuffle and sampler is None),
        persistent_workers=(num_workers > 0)
    )
    
    logger.info('CXR data loader created')
    
    return dataset, data_loader, sampler


# Utility function to get dataset statistics
def get_cxr_dataset_stats(root_path):
    """
    Get statistics about the CXR datasets without loading images.
    
    Returns dict with counts per dataset and total.
    """
    stats = {}
    total = 0
    
    dataset = CXRDataset(root_path=root_path, transform=None)
    
    from collections import Counter
    source_counts = Counter(dataset.dataset_sources)
    
    for source, count in source_counts.items():
        stats[source] = count
        total += count
    
    stats['total'] = total
    return stats


if __name__ == '__main__':
    # Quick test
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Update this path to your dataset location
    root = "Datasets/CXR"
    
    # Test dataset creation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    dataset = CXRDataset(root_path=root, transform=transform)
    print(f"\nTotal images: {len(dataset)}")
    
    # Test loading a few images
    for i in range(min(3, len(dataset))):
        img = dataset[i]
        print(f"Image {i}: shape={img.shape}, dtype={img.dtype}")
