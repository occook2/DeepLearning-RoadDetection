"""
Download and setup datasets for CNN multitask detection project.

This script downloads:
- SuperTuxKart Classification Dataset (for CNN classifier training)
- SuperTuxKart Drive Dataset (for detection/segmentation training)
"""

import urllib.request
import zipfile
from pathlib import Path


def download_dataset(url: str, zip_name: str, project_dir: Path):
    """Download and extract a dataset."""
    zip_path = project_dir / zip_name
    
    print(f"Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"✓ Downloaded to {zip_path}")
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        return False
    
    print(f"Extracting {zip_name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(project_dir)
        print(f"✓ Extracted successfully")
    except Exception as e:
        print(f"✗ Error extracting: {e}")
        return False
    
    try:
        zip_path.unlink()
        print("✓ Cleaned up zip file")
    except Exception as e:
        print(f"Warning: Could not delete zip file: {e}")
    
    return True


def main():
    """Download both required datasets."""
    project_dir = Path(__file__).parent
    
    print("=" * 60)
    print("CNN Multi-Task Detection - Dataset Download")
    print("=" * 60)
    
    # Dataset 1: Classification Data
    print("\n[1/2] SuperTuxKart Classification Dataset")
    print("-" * 60)
    classification_url = "https://www.cs.utexas.edu/~bzhou/dl_class/classification_data.zip"
    success1 = download_dataset(classification_url, "classification_data.zip", project_dir)
    
    # Dataset 2: Drive Data
    print("\n[2/2] SuperTuxKart Drive Dataset")
    print("-" * 60)
    drive_url = "https://www.cs.utexas.edu/~bzhou/dl_class/drive_data.zip"
    success2 = download_dataset(drive_url, "drive_data.zip", project_dir)
    
    # Verify extraction
    print("\n" + "=" * 60)
    if success1 and success2:
        classification_dir = project_dir / "classification_data"
        drive_dir = project_dir / "drive_data"
        
        if classification_dir.exists() and drive_dir.exists():
            print("✓ All datasets ready!")
            print(f"  Classification: {classification_dir}")
            print(f"  Drive Data: {drive_dir}")
            print("\nYou can now run training scripts:")
            print("  cd src")
            print("  python train_classification.py --num_epoch 50")
            print("  python train_detection.py --num_epoch 100")
        else:
            print("✗ Dataset directories not found after extraction")
    else:
        print("✗ Dataset download failed")
        print("\nPlease manually download datasets from:")
        print("  Classification: https://www.cs.utexas.edu/~bzhou/dl_class/classification_data.zip")
        print("  Drive: https://www.cs.utexas.edu/~bzhou/dl_class/drive_data.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
