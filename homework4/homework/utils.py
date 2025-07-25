import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .datasets.road_dataset import load_data as load_road_data

LABEL_NAMES = ["background", "kart", "pickup", "nitro", "bomb", "projectile"]


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path: str):
        """
        Pairs of images and labels (int) for classification
        You won't need to modify this, but all PyTorch datasets must implement these methods
        """
        to_tensor = transforms.ToTensor()

        self.data = []

        with open(Path(dataset_path, "labels.csv"), newline="") as f:
            for fname, label, _ in csv.reader(f):
                if label in LABEL_NAMES:
                    image = Image.open(Path(dataset_path, fname))
                    label_id = LABEL_NAMES.index(label)

                    self.data.append((to_tensor(image), label_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(
        dataset_path: str, 
        task: str = 'classifier', 
        transform_pipeline: str = 'state_only', 
        num_workers: int = 0, 
        batch_size: int = 128, 
        shuffle: bool = False
        ) -> DataLoader:
    
    if task == "planner":
        return load_road_data(
            dataset_path = dataset_path,
            transform_pipeline = transform_pipeline,
            return_dataloader = True,
            num_workers = num_workers,
            batch_size = batch_size,
            shuffle = shuffle
        )
    elif task == "classifer":
        dataset = SuperTuxDataset(dataset_path)
        return DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    else:
        raise ValueError(f'Unknown task type: {task}')


def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor):
    """
    Arguments:
        outputs: torch.Tensor, shape (b, num_classes) either logits or probabilities
        labels: torch.Tensor, shape (b,) with the ground truth class labels

    Returns:
        a single torch.Tensor scalar
    """
    outputs_idx = outputs.max(1)[1].type_as(labels)

    return (outputs_idx == labels).float().mean()
