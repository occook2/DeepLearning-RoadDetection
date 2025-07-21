import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from .models import Classifier, load_model, save_model
from .datasets import road_dataset
from .metrics import ConfusionMatrix


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
    log_dir = Path(exp_dir) / f"detection_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Use kwargs for grader. Go into train mode for model
    model = load_model("detector", **kwargs)
    model = model.to(device)
    print(model)

    # Load train and val data. Check time this takes
    print(f'Loading started')
    load_start = time.time()
    train_data = road_dataset.load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=0)
    val_data = road_dataset.load_data("drive_data/val", shuffle=False, batch_size=batch_size, num_workers=0)
    load_end = time.time()
    print(f'Total loading time: {(load_end - load_start):.2f} sec')

    # create loss functions and optimizer
    seg_loss = nn.CrossEntropyLoss()
    depth_loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    global_step = 0
    metrics = {"train_seg_acc": [], "train_seg_iou": [],"train_depth_mae": [], "val_seg_acc": [], "val_seg_iou": [], "val_depth_mae": []}
    confmat_train = ConfusionMatrix(num_classes=3)
    confmat_val = ConfusionMatrix(num_classes=3)
    
    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()
        confmat_train.reset()
        confmat_val.reset()


        model.train()

        # Setting Up Timings to Determine Bottlenecks
        total_epoch_start = time.time()
        data_loading_time = 0.0
        model_compute_time = 0.0
        data_iter_start = time.time()

        for batch in train_data:
            # Unpack batch of images
            img = batch["image"].to(device, non_blocking=True)
            target_track = batch["track"].to(device, non_blocking=True)
            target_depth = batch["depth"].to(device, non_blocking=True)
            if target_depth.dim() == 3:
                target_depth = target_depth.unsqueeze(dim=1)

            batch_data_loaded = time.time()
            data_loading_time += batch_data_loaded - data_iter_start
            
            batch_compute_start = time.time()
            
            # Get Model Prediction
            seg_logits, pred_depth = model(img)
            
            # Calculate Loss
            seg_loss_val = seg_loss(seg_logits, target_track)
            depth_loss_val = depth_loss(pred_depth, target_depth)
            loss_val = seg_loss_val + lambda_depth * depth_loss_val

            # Backward Propagation
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Save predictions
            seg_pred = seg_logits.argmax(dim=1)
            confmat_train.add(seg_pred.flatten(), target_track.flatten())
            
            # Save Training Metrics
            train_metrics = confmat_train.compute()
            metrics["train_seg_acc"].append(train_metrics["accuracy"])
            metrics["train_seg_iou"].append(train_metrics["iou"])
            
            # Depth Accuracy
            depth_mae = torch.abs(pred_depth - target_depth).mean().item()
            metrics["train_depth_mae"].append(depth_mae)
            
            
            global_step += 1

            model_compute_time += time.time() - batch_compute_start
            # Mark the start of the next data load
            data_iter_start = time.time()
        
        total_epoch_time = time.time() - total_epoch_start

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                img = batch["image"].to(device, non_blocking=True)
                target_track = batch["track"].to(device, non_blocking=True)
                target_depth = batch["depth"].to(device, non_blocking=True)

                # TODO: compute validation accuracy
                seg_logits, pred_depth = model(img)
                
                # Segmentation Accuracy
                seg_pred = seg_logits.argmax(dim=1)
                confmat_val.add(seg_pred.flatten(), target_track.flatten())

                val_metrics = confmat_val.compute()
                metrics["val_seg_acc"].append(val_metrics["accuracy"])
                metrics["val_seg_iou"].append(val_metrics["iou"])

                # Depth MAE
                depth_mae = torch.abs(pred_depth - target_depth).mean().item()
                metrics["val_depth_mae"].append(depth_mae)

        # log average train and val accuracy to tensorboard
        epoch_train_seg_acc = torch.as_tensor(metrics["train_seg_acc"]).mean()
        epoch_val_seg_acc = torch.as_tensor(metrics["val_seg_acc"]).mean()
        epoch_train_seg_iou = torch.as_tensor(metrics["train_seg_iou"]).mean()
        epoch_val_seg_iou = torch.as_tensor(metrics["val_seg_iou"]).mean()
        epoch_train_depth_mae = torch.as_tensor(metrics["train_depth_mae"]).mean()
        epoch_val_depth_mae = torch.as_tensor(metrics["val_depth_mae"]).mean()


        logger.add_scalar('train/segmentation_accuracy', epoch_train_seg_acc, global_step)
        logger.add_scalar('train/mIoU', epoch_train_seg_iou, global_step)
        logger.add_scalar('train/depth_mae', epoch_train_depth_mae, global_step)
        logger.add_scalar('val/segmentation_accuracy', epoch_val_seg_acc, global_step)
        logger.add_scalar('val/mIoU', epoch_val_seg_iou, global_step)
        logger.add_scalar('val/depth_mae', epoch_val_depth_mae, global_step)

        # Print information for each epoch
        print(f"Epoch {epoch + 1:2d} / {num_epoch:2d}")
        print(f"  Train Segmentation Accuracy : {epoch_train_seg_acc:.4f}")
        print(f"  Val   Segmentation Accuracy : {epoch_val_seg_acc:.4f}")
        print(f"  Train Segmentation mIoU     : {epoch_train_seg_iou:.4f}")
        print(f"  Val   Segmentation mIoU     : {epoch_val_seg_iou:.4f}")
        print(f"  Train Depth MAE             : {epoch_train_depth_mae:.4f}")
        print(f"  Val   Depth MAE             : {epoch_val_depth_mae:.4f}")


        # print(f"  mIoU train: {train_iou:.4f} val: {val_iou:.4f}")
        print(f"  Total epoch time      : {total_epoch_time:.2f} sec")
        print(f"  Data loading time     : {data_loading_time:.2f} sec")
        print(f"  Model compute time    : {model_compute_time:.2f} sec")
        print(f"  Remaining (overhead?) : {total_epoch_time - data_loading_time - model_compute_time:.2f} sec")
    
    # Save model for grader
    save_model(model)

    # Optional: also save a copy to your log directory
    torch.save(model.state_dict(), log_dir / "detector.th")
    print(f"Model saved to {log_dir / 'detector.th'}")

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