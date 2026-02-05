"""
Create PyTorch DataLoaders for training
"""

import torch
from torch.utils.data import DataLoader
from .dataset import ChestXrayDataset
from .transforms import get_train_transforms, get_val_transforms

def get_dataloaders(data_dir='data/processed', 
                    batch_size=32, 
                    img_size=224,
                    num_workers=4):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir (str): Path to processed data directory
        batch_size (int): Batch size for training
        img_size (int): Image size (224 for ResNet, 380 for EfficientNet)
        num_workers (int): Number of workers for data loading
        
    Returns:
        dict: Dictionary containing train_loader, val_loader, test_loader
    """
    
    # Get transforms
    train_transform = get_train_transforms(img_size)
    val_transform = get_val_transforms(img_size)
    
    # Create datasets
    train_dataset = ChestXrayDataset(f'{data_dir}/train', transform=train_transform)
    val_dataset = ChestXrayDataset(f'{data_dir}/val', transform=val_transform)
    test_dataset = ChestXrayDataset(f'{data_dir}/test', transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Print dataset info
    print("="*60)
    print("DATALOADER SUMMARY")
    print("="*60)
    print(f"\nImage size: {img_size}×{img_size}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    
    print(f"\nTrain dataset:")
    print(f"  Total images: {len(train_dataset)}")
    print(f"  Batches: {len(train_loader)}")
    print(f"  Class distribution: {train_dataset.get_class_distribution()}")
    
    print(f"\nValidation dataset:")
    print(f"  Total images: {len(val_dataset)}")
    print(f"  Batches: {len(val_loader)}")
    print(f"  Class distribution: {val_dataset.get_class_distribution()}")
    
    print(f"\nTest dataset:")
    print(f"  Total images: {len(test_dataset)}")
    print(f"  Batches: {len(test_loader)}")
    print(f"  Class distribution: {test_dataset.get_class_distribution()}")
    
    print("\n" + "="*60)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

# Test dataloaders
if __name__ == "__main__":
    # Get dataloaders
    dataloaders = get_dataloaders(batch_size=32, img_size=224, num_workers=0)
    
    # Test a batch
    train_loader = dataloaders['train']
    images, labels = next(iter(train_loader))
    
    print(f"\nSample batch:")
    print(f"  Images shape: {images.shape}")  # Should be [32, 3, 224, 224]
    print(f"  Labels shape: {labels.shape}")  # Should be [32]
    print(f"  Label distribution in batch: {torch.bincount(labels)}")
    
    print("\n✅ DataLoaders working correctly!")