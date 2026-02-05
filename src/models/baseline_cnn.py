"""
Baseline CNN Model - Simple 3-layer ConvNet
"""

import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    """
    Simple CNN for chest X-ray classification
    
    Architecture:
        - 3 Convolutional blocks (Conv → ReLU → MaxPool)
        - 2 Fully connected layers
        - Binary classification output
    """
    
    def __init__(self, num_classes=1):
        super(BaselineCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(
            in_channels=3,      # RGB input
            out_channels=32,    # 32 filters
            kernel_size=3,      # 3×3 kernel
            padding=1           # Keep spatial dimensions
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224×224 → 112×112
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 112×112 → 56×56
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  # 56×56 → 28×28
        
        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # 128 channels × 28 × 28
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input images [batch_size, 3, 224, 224]
            
        Returns:
            torch.Tensor: Logits [batch_size, 1]
        """
        # Conv Block 1
        x = self.conv1(x)      # [B, 32, 224, 224]
        x = self.relu1(x)
        x = self.pool1(x)      # [B, 32, 112, 112]
        
        # Conv Block 2
        x = self.conv2(x)      # [B, 64, 112, 112]
        x = self.relu2(x)
        x = self.pool2(x)      # [B, 64, 56, 56]
        
        # Conv Block 3
        x = self.conv3(x)      # [B, 128, 56, 56]
        x = self.relu3(x)
        x = self.pool3(x)      # [B, 128, 28, 28]
        
        # Fully Connected
        x = self.flatten(x)    # [B, 128*28*28] = [B, 100352]
        x = self.fc1(x)        # [B, 256]
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)        # [B, 1]
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Test the model
if __name__ == "__main__":
    # Create model
    model = BaselineCNN(num_classes=1)
    
    # Print model architecture
    print("="*60)
    print("BASELINE CNN ARCHITECTURE")
    print("="*60)
    print(model)
    
    # Count parameters
    num_params = model.count_parameters()
    print("\n" + "="*60)
    print(f"Total trainable parameters: {num_params:,}")
    print("="*60)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    output = model(dummy_input)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(output)
    print(f"  Probabilities: {probs.squeeze()}")
    
    print("\n✅ Model architecture working correctly!")