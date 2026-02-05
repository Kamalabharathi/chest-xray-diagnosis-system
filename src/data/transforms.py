"""
Data transforms and augmentation for training
"""
import torch

from torchvision import transforms

def get_train_transforms(img_size=224):
    """
    Get training transforms with data augmentation
    
    Args:
        img_size (int): Target image size
        
    Returns:
        transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        
        # Data augmentation
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        
        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                           std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(img_size=224):
    """
    Get validation/test transforms (no augmentation)
    
    Args:
        img_size (int): Target image size
        
    Returns:
        transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

# Test transforms
if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load a sample image
    img_path = "data/processed/test/PNEUMONIA/person1_virus_6.jpeg"
    img = Image.open(img_path).convert('RGB')
    
    # Apply train transforms (with augmentation)
    train_transform = get_train_transforms(224)
    
    # Create figure with multiple augmented versions
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Apply transform 7 times to see different augmentations
    for i in range(1, 8):
        row = i // 4
        col = i % 4
        
        # Apply transform
        img_tensor = train_transform(img)
        
        # Denormalize for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img_tensor * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        
        # Convert to numpy and show
        img_np = img_denorm.permute(1, 2, 0).numpy()
        axes[row, col].imshow(img_np)
        axes[row, col].set_title(f'Augmented {i}')
        axes[row, col].axis('off')
    
    plt.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/augmentation_examples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Transforms working correctly!")
    print("✅ Augmentation examples saved to: results/plots/augmentation_examples.png")