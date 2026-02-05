"""
Custom PyTorch Dataset for Chest X-Ray Images
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import numpy as np

class ChestXrayDataset(Dataset):
    """
    Custom Dataset for Chest X-Ray images
    
    Args:
        data_dir (str): Path to data directory (e.g., 'data/processed/train')
        transform (callable, optional): Optional transform to be applied on images
    """
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Get all image paths
        self.image_paths = []
        self.labels = []
        
        # Normal = 0, Pneumonia = 1
        for label, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.glob('*.jpeg'):
                self.image_paths.append(img_path)
                self.labels.append(label)
        
        # Convert to numpy arrays
        self.labels = np.array(self.labels)
        
        # Calculate class weights for imbalance handling
        self.class_counts = np.bincount(self.labels)
        self.class_weights = 1.0 / self.class_counts
        self.class_weights = self.class_weights / self.class_weights.sum()
    
    def __len__(self):
        """Return the total number of images"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a single image and label
        
        Args:
            idx (int): Index of the image
            
        Returns:
            image (torch.Tensor): Transformed image
            label (int): Class label (0 = Normal, 1 = Pneumonia)
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB (3 channels)
        
        # Get label
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """Return class distribution"""
        return {
            'NORMAL': int(self.class_counts[0]),
            'PNEUMONIA': int(self.class_counts[1]),
            'weights': self.class_weights.tolist()
        }

# Test the dataset
if __name__ == "__main__":
    from torchvision import transforms
    
    # Simple transform for testing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Load dataset
    dataset = ChestXrayDataset('data/processed/train', transform=transform)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Get a sample
    image, label = dataset[0]
    print(f"\nSample image shape: {image.shape}")
    print(f"Sample label: {label} ({'NORMAL' if label == 0 else 'PNEUMONIA'})")
    
    print("\n✅ Dataset class working correctly!")