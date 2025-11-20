import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from .models import MLPPlanner, load_model, save_model
from .utils import load_data

if __name__ == "__main__":
    # Hardware Selection
    if torch.cuda.is_available():
        print("CUDA available, using GPU")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seed so each run is deterministic
    torch.manual_seed(2024)
    np.random.seed(2024)

    train_data = load_data("drive_data/train", task="planner", batch_size=256, shuffle=True)
    all_points = []

    for sample in train_data:
        track_left = sample["track_left"]  # (B, 10, 2)
        track_right = sample["track_right"]  # (B, 10, 2)

        # Concatenate and flatten into (B * 20, 2)
        combined = torch.cat([track_left, track_right], dim=1).view(-1, 2)
        all_points.append(combined)

    # Concatenate across batches → shape (N, 2)
    all_points_tensor = torch.cat(all_points, dim=0)

    # Compute mean and std across x and y (dim=0)
    mean = all_points_tensor.mean(dim=0)
    std = all_points_tensor.std(dim=0)

    print(f"Mean (x, y): {mean.tolist()}")
    print(f"Std (x, y): {std.tolist()}")


