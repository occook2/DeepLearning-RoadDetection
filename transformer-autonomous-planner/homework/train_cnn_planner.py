import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets import road_dataset

from .utils import load_data


def train(
    exp_dir: str = "logs",
    num_epoch: int = 25,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    lambda_depth: float = 0.5,
    **kwargs,
):
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
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"cnn_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Use kwargs for grader. Go into train mode for model
    model = load_model("cnn_planner", **kwargs)
    model = model.to(device)
    model.train()

    # Load Data
    train_data = road_dataset.load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=0)
    val_data = road_dataset.load_data("drive_data/val", shuffle=False, batch_size=batch_size, num_workers=0)

    train_data_target = load_data("drive_data/train", task="planner", batch_size=batch_size, shuffle=True)
    val_data_target = load_data("drive_data/val", task="planner", batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    global_step = 0
    metrics = {"train_loss": [], "train_long_error": [], "train_lat_error": [], "val_loss": [], "val_long_error": [], "val_lat_error": [],}

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for sample_img, sample_target in zip(train_data, train_data_target):
            img = sample_img["image"].to(device, non_blocking=True)
            waypoints = sample_target["waypoints"].to(device, non_blocking=True)
            waypoints_mask = sample_target["waypoints_mask"].to(device, non_blocking=True)

            # Predict and calculate loss
            pred_waypoints = model(img)
            planner_loss = (pred_waypoints - waypoints).abs() * waypoints_mask[..., None]
            loss_val = planner_loss.sum() / waypoints_mask.sum()

            # Optimize
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Log metrics
            metrics["train_loss"].append(loss_val.item())
            global_step += 1

        with torch.inference_mode():
            model.eval()

            for sample_img, sample_target in zip(val_data, val_data_target):
                img = sample_img["image"].to(device, non_blocking=True)
                waypoints = sample_target["waypoints"].to(device, non_blocking=True)
                waypoints_mask = sample_target["waypoints_mask"].to(device, non_blocking=True)

                pred = model(img)
                error = (pred - waypoints).abs() * waypoints_mask[..., None]
                long_error = error[:, :, 0].sum() / waypoints_mask.sum()
                lat_error = error[:, :, 1].sum() / waypoints_mask.sum()

                loss_val = ((pred - waypoints) ** 2).sum(dim=-1)
                loss_val = loss_val * waypoints_mask
                loss_val = loss_val.sum() / waypoints_mask.sum()

                metrics["val_loss"].append(loss_val.item())
                metrics["val_long_error"].append(long_error.item())
                metrics["val_lat_error"].append(lat_error.item())

        epoch_train_loss = torch.as_tensor(metrics["train_loss"]).mean()
        epoch_val_loss = torch.as_tensor(metrics["val_loss"]).mean()
        epoch_val_long_error = torch.as_tensor(metrics["val_long_error"]).mean()
        epoch_val_lat_error = torch.as_tensor(metrics["val_lat_error"]).mean()

        logger.add_scalar('train/loss', sum(metrics["train_loss"]) / len(metrics["train_loss"]), global_step)
        logger.add_scalar('val/loss', sum(metrics["val_loss"]) / len(metrics["val_loss"]), global_step)
        logger.add_scalar('val/long_error', sum(metrics["val_long_error"]) / len(metrics["val_long_error"]), global_step)
        logger.add_scalar('val/lat_error', sum(metrics["val_lat_error"]) / len(metrics["val_lat_error"]), global_step)

        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: \n"
            f"train_loss={epoch_train_loss:.4f} \n"
            f"val_loss={epoch_val_loss:.4f} \n"
            f"val_long_loss={epoch_val_long_error}\n"
            f"val_lat_loss={epoch_val_lat_error}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = model 
    
    save_model(best_model)
    print(f"Model saved to {log_dir / 'cnn_planner.th'}") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_depth", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))