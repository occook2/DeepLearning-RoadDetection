"""
Download and setup the SuperTuxKart drive dataset.

This script downloads the drive dataset from the course website and extracts it
to the correct location (parent directory) for training the autonomous planner.
"""

import urllib.request
import zipfile
from pathlib import Path


def download_dataset():
    """Download and extract the SuperTuxKart drive dataset."""
    
    # Dataset URL
    dataset_url = "https://www.cs.utexas.edu/~bzhou/dl_class/drive_data.zip"
    
    # Determine paths - dataset goes in parent directory
    src_dir = Path(__file__).parent
    project_dir = src_dir.parent
    zip_path = project_dir / "drive_data.zip"
    
    print("=" * 60)
    print("Transformer Autonomous Planner - Dataset Download")
    print("=" * 60)
    print(f"\nDownloading SuperTuxKart Drive Dataset...")
    print(f"URL: {dataset_url}")
    print(f"Destination: {project_dir}")
    
    # Download the dataset
    try:
        urllib.request.urlretrieve(dataset_url, zip_path)
        print(f"✓ Downloaded to {zip_path}")
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        return False
    
    # Extract the dataset
    print("\nExtracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(project_dir)
        print(f"✓ Extracted to {project_dir}")
    except Exception as e:
        print(f"✗ Error extracting dataset: {e}")
        return False
    
    # Clean up zip file
    try:
        zip_path.unlink()
        print("✓ Cleaned up zip file")
    except Exception as e:
        print(f"Warning: Could not delete zip file: {e}")
    
    # Verify extraction
    drive_dir = project_dir / "drive_data"
    train_dir = drive_dir / "train"
    val_dir = drive_dir / "val"
    
    print("\n" + "=" * 60)
    if drive_dir.exists() and train_dir.exists() and val_dir.exists():
        print("✓ Dataset ready!")
        print(f"  Drive Data: {drive_dir}")
        print(f"  Train: {train_dir}")
        print(f"  Val: {val_dir}")
        print("\nYou can now run training scripts:")
        print("  cd src")
        print("  python train_planner.py --num_epoch 50")
        print("  python train_transformer.py --num_epoch 50")
        print("  python train_cnn_planner.py --num_epoch 50")
        print("=" * 60)
        return True
    else:
        print("✗ Dataset extraction may have failed. Expected directories not found.")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = download_dataset()
    
    if not success:
        print("\nPlease manually download the dataset from:")
        print("  https://www.cs.utexas.edu/~bzhou/dl_class/drive_data.zip")
        print("And extract it to the transformer-autonomous-planner directory")
