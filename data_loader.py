"""
Data loading utilities for multimodal PCOS prediction.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MultimodalPCOSDataset(Dataset):
    """
    Dataset loader for PCOS prediction using ultrasound images and tabular data.
    
    Args:
        data_dir: Base directory containing ultrasound/ and tabular/ subdirectories
        split: 'train' or 'test' or 'val'
        img_size: Target image size (default: 64)
        load_real_data: If True, loads your data; if False, generates synthetic data
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        split: str = "train",
        img_size: int = 64,
        load_real_data: bool = True,
        use_processed_tabular: bool = False,
        processed_tabular_dir: str = "data/tabular_processed"
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.use_processed_tabular = use_processed_tabular
        self.processed_tabular_dir = Path(processed_tabular_dir)
        
        if load_real_data:
            self._load_real_data()
        else:
            # Generate synthetic data for testing
            self._generate_synthetic_data()
    
    def _load_real_data(self):
        """Load real data; optionally use preprocessed tabular arrays."""
        tabular_csv_path = self.data_dir / "tabular" / f"{self.split}.csv"
        if not tabular_csv_path.exists():
            raise FileNotFoundError(
                f"Please create a {self.split}.csv file in data/tabular/\n"
                "Expected format: CSV with feature columns + 'label' column (0/1 for PCOS)"
            )
        df = pd.read_csv(tabular_csv_path)
        if 'label' not in df.columns:
            raise ValueError("CSV must have a 'label' column (0/1 for PCOS)")
        # Default from CSV
        labels = df['label'].values.astype(np.int64)
        feature_cols = [c for c in df.columns if c != 'label']
        tabular = df[feature_cols].values.astype(np.float32)
        # Optionally override with processed arrays
        if self.use_processed_tabular:
            x_path = self.processed_tabular_dir / f"{self.split}.npy"
            y_path = self.processed_tabular_dir / f"labels_{self.split}.npy"
            if not x_path.exists() or not y_path.exists():
                raise FileNotFoundError(
                    f"Processed arrays not found for split '{self.split}' in {self.processed_tabular_dir}.\n"
                    "Run: python preprocess_data.py --data_dir data --out_dir data/tabular_processed"
                )
            tabular = np.load(x_path)
            labels = np.load(y_path).astype(np.int64)
        self.labels = labels
        self.tabular = tabular.astype(np.float32)
        self.n_samples = len(self.labels)
        
        # Load ultrasound images
        self.ultrasound_paths = self._get_image_paths("ultrasound")
        
        # Log data sizes (informational only)
        print(f"Found {len(self.ultrasound_paths)} ultrasound images")
        print(f"  • {len(self.label_to_images[1])} PCOS/infected")
        print(f"  • {len(self.label_to_images[0])} healthy/non-infected")
        print(f"Using {len(self.labels)} tabular samples")
        
        # Image preprocessing
        self.ultrasound_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
    
    def _get_image_paths(self, modality: str) -> List[Path]:
        """Get paths to images from infected/noninfected folders."""
        img_dir = self.data_dir / modality / self.split
        
        if not img_dir.exists():
            raise FileNotFoundError(
                f"Please create a {self.split}/ folder in data/{modality}/\n"
                f"Expected structure: data/{modality}/{self.split}/"
            )
        
        # Supported image formats
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm'}
        infected_paths = []
        noninfected_paths = []
        
        # Check both 'infected' and 'non infected' folders
        infected_dir = img_dir / 'infected'
        noninfected_dir = img_dir / 'non infected'
        
        # Get images from infected folder (PCOS positive cases)
        if infected_dir.exists():
            for img_file in infected_dir.iterdir():
                if img_file.suffix.lower() in extensions:
                    infected_paths.append(img_file)
        
        # Get images from non-infected folder (PCOS negative cases)
        if noninfected_dir.exists():
            for img_file in noninfected_dir.iterdir():
                if img_file.suffix.lower() in extensions:
                    noninfected_paths.append(img_file)
        
        if not infected_paths and not noninfected_paths:
            raise FileNotFoundError(
                f"No images found in {img_dir}/infected or {img_dir}/non infected\n"
                f"Supported formats: {extensions}"
            )
        
        # Sort for consistency
        infected_paths.sort()
        noninfected_paths.sort()
        
        # Create mapping: label -> image paths
        self.label_to_images = {
            1: infected_paths,    # PCOS positive
            0: noninfected_paths  # PCOS negative
        }
        
        # For each tabular sample with label l, we'll randomly select
        # an image from label_to_images[l] when __getitem__ is called
        return infected_paths + noninfected_paths
    
    def _generate_synthetic_data(self, n_samples: int = 200):
        """Generate synthetic data for testing (when real data not available)."""
        self.n_samples = n_samples
        self.ultrasounds = np.random.rand(n_samples, 1, self.img_size, self.img_size).astype(np.float32)
        self.tabular = np.random.randn(n_samples, 10).astype(np.float32)
        
        # Generate synthetic labels
        logits = (
            self.tabular[:, 0] * 0.6 + 
            self.tabular[:, 1] * -0.3 + 
            np.mean(self.ultrasounds.reshape(n_samples, -1), axis=1) * 0.5
        )
        probs = 1 / (1 + np.exp(-logits))
        self.labels = (probs > 0.5).astype(np.int64)
        
        self.ultrasound_paths = None
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int):
        """Return ultrasound image, tabular features, and label for one sample."""
        # Handle tabular data
        tab = torch.tensor(self.tabular[idx])
        label = torch.tensor(int(self.labels[idx]))
        
        # Handle ultrasound image
        if hasattr(self, 'label_to_images'):
            # Real data: randomly select an image with matching label
            label_val = int(label.item())
            matching_images = self.label_to_images[label_val]
            if not matching_images:
                # If no images for this label, use any available image
                matching_images = self.ultrasound_paths
            
            # Use deterministic random selection based on idx
            img_idx = hash(str(idx)) % len(matching_images)
            img_path = matching_images[img_idx]
            
            us_img = Image.open(img_path).convert('L')
            us_tensor = self.ultrasound_transform(us_img)
            return us_tensor, tab, label
        else:
            # Synthetic data
            us_tensor = torch.tensor(self.ultrasounds[idx])
            return us_tensor, tab, label


def split_data(data_dir: str = "data", train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Split your data into train/val/test sets.
    
    This is a helper function. You should manually organize your data or
    call this function once to split it.
    
    Args:
        data_dir: Directory containing all data files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
    """
    import shutil
    from sklearn.model_selection import train_test_split
    
    data_dir = Path(data_dir)
    tabular_file = data_dir / "tabular" / "all_data.csv"
    
    if not tabular_file.exists():
        raise FileNotFoundError(f"Please create data/tabular/all_data.csv first")
    
    df = pd.read_csv(tabular_file)
    
    # Split indices
    train_idx, temp_idx = train_test_split(
        range(len(df)), 
        test_size=(1 - train_ratio),
        random_state=42
    )
    
    val_size = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_size),
        random_state=42
    )
    
    # Save splits
    for split, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_df = df.iloc[idx].reset_index(drop=True)
        split_df.to_csv(data_dir / "tabular" / f"{split}.csv", index=False)
        print(f"Created {split}.csv with {len(idx)} samples")
    
    print("\n💡 Now organize your ultrasound and facial images into:")
    print("   data/ultrasound/train/, data/ultrasound/val/, data/ultrasound/test/")
    print("   data/facial/train/, data/facial/val/, data/facial/test/")


if __name__ == "__main__":
    # Example usage
    # Try loading real data first, fall back to synthetic if needed
    try:
        dataset = MultimodalPCOSDataset(
            data_dir="data",
            split="train",
            load_real_data=True,  # Try real data first
            img_size=64
        )
        print(f"\n✓ Loaded {len(dataset)} samples from real data")
        print(f"  Tabular features: {dataset.tabular.shape[1]}")
        print(f"  Total ultrasound images: {len(dataset.ultrasound_paths)}")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"\n⚠ Using synthetic data: {str(e)}")
        dataset = MultimodalPCOSDataset(
            data_dir="data",
            split="train",
            load_real_data=False,
            img_size=64
        )
        print(f"✓ Generated {len(dataset)} synthetic samples")
        print(f"  Tabular features: {dataset.tabular.shape[1]}")
    
    # Show first sample
    us, tab, label = dataset[0]
    print(f"\nFirst sample shapes:")
    print(f"  Ultrasound (grayscale): {us.shape}")  # [1, H, W]
    print(f"  Tabular features: {tab.shape}")       # [num_features]
    print(f"  Label: {label}")                      # 0 or 1

