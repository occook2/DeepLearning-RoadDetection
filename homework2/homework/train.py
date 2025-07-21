import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import ClassificationLoss, load_model, save_model
from .utils import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 25,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=4)
    val_data = load_data("classification_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = ClassificationLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

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

        for img, label in train_data:
            batch_data_loaded = time.time()
            data_loading_time += batch_data_loaded - data_iter_start
            
            img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)
            
            batch_compute_start = time.time()
            # TODO: implement training step
            pred = model(img)
            loss_val = loss_func(pred, label)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            acc = (pred.argmax(dim=1) == label).float().mean().item()
            metrics["train_acc"].append(acc)
            global_step += 1

            model_compute_time += time.time() - batch_compute_start
            # Mark the start of the next data load
            data_iter_start = time.time()
        
        total_epoch_time = time.time() - total_epoch_start

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for img, label in val_data:
                img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)

                # TODO: compute validation accuracy
                pred = model(img)
                acc = (pred.argmax(dim=1) == label).float().mean().item()
                metrics["val_acc"].append(acc)

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        logger.add_scalar('train/accuracy', sum(metrics["train_acc"]) / len(metrics["train_acc"]), global_step)
        logger.add_scalar('val/accuracy', sum(metrics["val_acc"]) / len(metrics["val_acc"]), global_step)
        logger.add_scalar('learning_rate', lr)
        logger.add_scalar('batch_size', batch_size)


        if sum(metrics["val_acc"]) / len(metrics["val_acc"]) > 0.82:
            break

        # print every epoch
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_acc={epoch_train_acc:.4f} "
            f"val_acc={epoch_val_acc:.4f}"
        )
        print(f"  Total epoch time      : {total_epoch_time:.2f} sec")
        print(f"  Data loading time     : {data_loading_time:.2f} sec")
        print(f"  Model compute time    : {model_compute_time:.2f} sec")
        print(f"  Remaining (overhead?) : {total_epoch_time - data_loading_time - model_compute_time:.2f} sec")


    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
