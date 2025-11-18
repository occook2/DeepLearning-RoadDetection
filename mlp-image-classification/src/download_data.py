"""
Download and setup the SuperTuxKart classification dataset.

This script downloads the dataset from the course website and extracts it
to the correct location for training.
"""

import urllib.request
import zipfile
from pathlib import Path


def download_dataset():
    """Download and extract the SuperTuxKart classification dataset."""
    
    # Dataset URL
    dataset_url = "https://www.cs.utexas.edu/~bzhou/dl_class/classification_data.zip"
    
    # Determine paths
    src_dir = Path(__file__).parent
    project_dir = src_dir.parent
    data_dir = project_dir / "data"
    zip_path = project_dir / "classification_data.zip"
    
    print("Downloading SuperTuxKart Classification Dataset...")
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
    train_dir = project_dir / "classification_data" / "train"
    val_dir = project_dir / "classification_data" / "val"
    
    if train_dir.exists() and val_dir.exists():
        print(f"\n✓ Dataset ready!")
        print(f"  Train: {train_dir}")
        print(f"  Val: {val_dir}")
        return True
    else:
        print("\n✗ Dataset extraction may have failed. Expected directories not found.")
        return False


if __name__ == "__main__":
    success = download_dataset()
    
    if success:
        print("\nYou can now run training with:")
        print("  python train.py --model_name mlp --num_epoch 50")
    else:
        print("\nPlease manually download the dataset from:")
        print("  https://www.cs.utexas.edu/~bzhou/dl_class/classification_data.zip")
        print("And extract it to the mlp-image-classification directory")
