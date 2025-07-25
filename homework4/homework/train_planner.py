# Copied training file from HW 3

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


def train(
    exp_dir: str = "logs",
    num_epoch: int = 25,
    lr: float = 1e-3,
    batch_size: int = 256, # Check batch_size
    seed: int = 2024,
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
    log_dir = Path(exp_dir) / f"mlp_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Use kwargs for grader. Go into train mode for model
    model = load_model("mlp_planner", **kwargs)
    model = model.to(device)
    model.train()

    # Load train and val data. Check time this takes
    print(f'Loading started')
    load_start = time.time()
    train_data = load_data("drive_data/train", task="planner", batch_size=batch_size, shuffle=True)
    val_data = load_data("drive_data/val", task="planner", batch_size=batch_size, shuffle=False)
    load_end = time.time()
    print(f'Total loading time: {(load_end - load_start):.2f} sec')

    # create loss function and optimizer
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    global_step = 0
    metrics = {"train_loss": [], "val_loss": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        # Setting Up Timings to Determine Bottlenecks
        total_epoch_start = time.time()
        data_loading_time = 0.0
        model_compute_time = 0.0
        data_iter_start = time.time()

        for sample in train_data:
            batch_data_loaded = time.time()
            data_loading_time += batch_data_loaded - data_iter_start
            
            track_left = sample["track_left"].to(device, non_blocking=True)
            track_right = sample["track_right"].to(device, non_blocking=True)
            waypoints = sample["waypoints"].to(device, non_blocking=True)
            waypoints_mask = sample["waypoints_mask"].to(device, non_blocking=True)
            
            batch_compute_start = time.time()
            
            # implement training step
            pred = model(track_left, track_right)
            
            # Manually calculating loss value to use waypoint mask - can convert to loss function if I don't want to mask
            loss_val = ((pred - waypoints) ** 2).sum(dim=-1)  # (B, 3)
            loss_val = loss_val * waypoints_mask  # (B, 3), mask out invalid ones
            loss_val = loss_val.sum() / waypoints_mask.sum()  # scalar average


            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            metrics["train_loss"].append(loss_val.item())
            global_step += 1

            model_compute_time += time.time() - batch_compute_start
            # Mark the start of the next data load
            data_iter_start = time.time()
        
        total_epoch_time = time.time() - total_epoch_start

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for sample in val_data:
                track_left = sample["track_left"].to(device, non_blocking=True)
                track_right = sample["track_right"].to(device, non_blocking=True)
                waypoints = sample["waypoints"].to(device, non_blocking=True)
                waypoints_mask = sample["waypoints_mask"].to(device, non_blocking=True)

                # TODO: compute validation accuracy
                pred = model(track_left, track_right)

                loss_val = ((pred - waypoints) ** 2).sum(dim=-1)
                loss_val = loss_val * waypoints_mask
                loss_val = loss_val.sum() / waypoints_mask.sum()

                metrics["val_loss"].append(loss_val.item())

        # log average train and val accuracy to tensorboard
        epoch_train_loss = torch.as_tensor(metrics["train_loss"]).mean()
        epoch_val_loss = torch.as_tensor(metrics["val_loss"]).mean()

        logger.add_scalar('train/loss', sum(metrics["train_loss"]) / len(metrics["train_loss"]), global_step)
        logger.add_scalar('val/loss', sum(metrics["val_loss"]) / len(metrics["val_loss"]), global_step)

        # Print information for each epoch
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_loss={epoch_train_loss:.4f} "
            f"val_loss={epoch_val_loss:.4f}"
        )
        print(f"  Total epoch time      : {total_epoch_time:.2f} sec")
        print(f"  Data loading time     : {data_loading_time:.2f} sec")
        print(f"  Model compute time    : {model_compute_time:.2f} sec")
        print(f"  Remaining (overhead?) : {total_epoch_time - data_loading_time - model_compute_time:.2f} sec")
    
    # Save model for grader
    save_model(model)

    # Optional: also save a copy to your log directory
    torch.save(model.state_dict(), log_dir / "classifier.th")
    print(f"Model saved to {log_dir / 'classifier.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))