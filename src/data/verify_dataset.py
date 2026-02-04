"""
Quick dataset verification script
"""
import os
from pathlib import Path

# Dataset path
data_dir = Path("data/raw/chest_xray")

# Check if dataset exists
if not data_dir.exists():
    print("❌ Dataset not found! Please download and extract first.")
    exit()

print("✅ Dataset found!")
print("\nDataset Structure:")
print("=" * 50)

# Count images in each split
for split in ['train', 'test', 'val']:
    split_path = data_dir / split
    
    if not split_path.exists():
        print(f"⚠️  {split}/ folder not found")
        continue
    
    normal_count = len(list((split_path / 'NORMAL').glob('*.jpeg')))
    pneumonia_count = len(list((split_path / 'PNEUMONIA').glob('*.jpeg')))
    total = normal_count + pneumonia_count
    
    print(f"\n{split.upper()}:")
    print(f"  Normal:    {normal_count:,} images")
    print(f"  Pneumonia: {pneumonia_count:,} images")
    print(f"  Total:     {total:,} images")
    print(f"  Imbalance: {pneumonia_count/normal_count:.2f}:1")

print("\n" + "=" * 50)
print("✅ Dataset verification complete!")