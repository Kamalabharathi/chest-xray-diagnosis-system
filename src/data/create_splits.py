# HASE 3: DATA PREPROCESSING PIPELINE 
#Raw Images → Preprocessing → PyTorch Dataset → DataLoader → Model
#Re-split dataset (fix tiny validation set)
#Custom PyTorch Dataset class
#Transforms & augmentation
#DataLoaders (train/val/test)

# STEP 1: Re-split Dataset Properly
# Problem: Validation set only has 16 images (too small!)
# Solution: Combine train+val, then re-split as 70/15/15

"""
Re-split dataset into proper train/val/test sets
"""

import shutil
from pathlib import Path
import random
from tqdm import tqdm

# Set seed for reproducibility
random.seed(42)

# Paths
raw_dir = Path("data/raw/chest_xray")
processed_dir = Path("data/processed")

# Create processed directory structure
for split in ['train', 'val', 'test']:
    for label in ['NORMAL', 'PNEUMONIA']:
        (processed_dir / split / label).mkdir(parents=True, exist_ok=True)

print("="*60)
print("RE-SPLITTING DATASET")
print("="*60)

# Step 1: Collect all training images (we'll re-split train+val)
print("\n📂 Collecting images from train and val sets...")

all_train_images = {
    'NORMAL': [],
    'PNEUMONIA': []
}

# Get train images
for label in ['NORMAL', 'PNEUMONIA']:
    train_images = list((raw_dir / 'train' / label).glob('*.jpeg'))
    val_images = list((raw_dir / 'val' / label).glob('*.jpeg'))
    all_train_images[label] = train_images + val_images
    print(f"  {label}: {len(train_images)} (train) + {len(val_images)} (val) = {len(all_train_images[label])} total")

# Step 2: Split into train/val (70/15)
print("\n✂️  Splitting into train (70%) / val (15%)...")

for label in ['NORMAL', 'PNEUMONIA']:
    images = all_train_images[label]
    random.shuffle(images)
    
    # Calculate split points
    total = len(images)
    train_size = int(0.7 / 0.85 * total)  # 70% of 85% of data
    
    train_images = images[:train_size]
    val_images = images[train_size:]
    
    print(f"\n  {label}:")
    print(f"    Train: {len(train_images)} images ({len(train_images)/total*100:.1f}%)")
    print(f"    Val:   {len(val_images)} images ({len(val_images)/total*100:.1f}%)")
    
    # Copy to new directories
    print(f"    Copying train images...", end=" ")
    for img_path in tqdm(train_images, desc=f"Train {label}", leave=False):
        dest = processed_dir / 'train' / label / img_path.name
        shutil.copy2(img_path, dest)
    print("✅")
    
    print(f"    Copying val images...", end=" ")
    for img_path in tqdm(val_images, desc=f"Val {label}", leave=False):
        dest = processed_dir / 'val' / label / img_path.name
        shutil.copy2(img_path, dest)
    print("✅")

# Step 3: Copy test set as-is
print("\n📋 Copying test set...")

for label in ['NORMAL', 'PNEUMONIA']:
    test_images = list((raw_dir / 'test' / label).glob('*.jpeg'))
    print(f"  {label}: {len(test_images)} images")
    
    for img_path in tqdm(test_images, desc=f"Test {label}", leave=False):
        dest = processed_dir / 'test' / label / img_path.name
        shutil.copy2(img_path, dest)

# Step 4: Verify new splits
print("\n" + "="*60)
print("NEW SPLIT DISTRIBUTION")
print("="*60)

for split in ['train', 'val', 'test']:
    normal_count = len(list((processed_dir / split / 'NORMAL').glob('*.jpeg')))
    pneumonia_count = len(list((processed_dir / split / 'PNEUMONIA').glob('*.jpeg')))
    total = normal_count + pneumonia_count
    
    print(f"\n{split.upper()}:")
    print(f"  Normal:    {normal_count:,} ({normal_count/total*100:.1f}%)")
    print(f"  Pneumonia: {pneumonia_count:,} ({pneumonia_count/total*100:.1f}%)")
    print(f"  Total:     {total:,}")
    print(f"  Imbalance: {pneumonia_count/normal_count:.2f}:1")

print("\n" + "="*60)
print("✅ DATASET RE-SPLIT COMPLETE!")
print(f"✅ New dataset location: {processed_dir}")
print("="*60)