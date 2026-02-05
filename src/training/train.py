"""
Training script for baseline CNN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Returns:
        avg_loss (float): Average loss for the epoch
        accuracy (float): Training accuracy
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Progress bar
    pbar = tqdm(dataloader, desc='Training')
    
    for images, labels in pbar:
        # Move to device
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # [batch_size, 1]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        
        # Get predictions
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """
    Validate the model
    
    Returns:
        avg_loss (float): Average validation loss
        metrics (dict): Dictionary of metrics
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            # Move to device
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track metrics
            running_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = running_loss / len(dataloader)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs)
    }
    
    return avg_loss, metrics

def train_model(model, train_loader, val_loader, num_epochs=20, 
                learning_rate=1e-3, device='cuda', save_dir='models/saved_models'):
    """
    Complete training loop
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_epochs (int): Number of epochs
        learning_rate (float): Learning rate
        device (str): 'cuda' or 'cpu'
        save_dir (str): Directory to save model checkpoints
        
    Returns:
        history (dict): Training history
    """
    
    # Setup
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    best_val_auc = 0.0
    
    print("="*60)
    print("TRAINING BASELINE CNN")
    print("="*60)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {train_loader.batch_size}")
    print("="*60)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}")
        print(f"Val AUC:    {val_metrics['auc']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'history': history
            }, f'{save_dir}/baseline_cnn_best.pth')
            print(f"✅ Saved best model (AUC: {best_val_auc:.4f})")
    
    print("\n" + "="*60)
    print(f"TRAINING COMPLETE!")
    print(f"Best Val AUC: {best_val_auc:.4f}")
    print("="*60)
    
    return history

def plot_training_history(history, save_path='results/plots/baseline_training.png'):
    """Plot training history"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[0, 1].set_title('Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC-ROC
    axes[1, 0].plot(epochs, history['val_auc'], 'g-', label='Val AUC')
    axes[1, 0].set_title('AUC-ROC', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision, Recall, F1
    axes[1, 1].plot(epochs, history['val_precision'], label='Precision')
    axes[1, 1].plot(epochs, history['val_recall'], label='Recall')
    axes[1, 1].plot(epochs, history['val_f1'], label='F1-Score')
    axes[1, 1].set_title('Precision, Recall, F1', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Baseline CNN Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Training plots saved to: {save_path}")

if __name__ == "__main__":
    from src.data.dataloader import get_dataloaders
    from src.models.baseline_cnn import BaselineCNN
    
    # Get dataloaders
    dataloaders = get_dataloaders(batch_size=32, img_size=224, num_workers=0)
    
    # Create model
    model = BaselineCNN(num_classes=1)
    
    # Train
    history = train_model(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        num_epochs=20,
        learning_rate=1e-3,
        device='cuda'
    )
    
    # Plot history
    plot_training_history(history)