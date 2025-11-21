from typing import Tuple, Dict, List

import os
import toml
import argparse

import torch
import numpy as np
import torch.optim as optim

from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    print("`wandb` was not found. `wandb` functionalities will be disabled.")


from mesh_opformer.dataset import MeshMapDataset
from mesh_opformer.dataset import collate_batch_w_sparse_fn
from mesh_opformer.dataset.utils import parse_segmentation_human_simplified_label

from mesh_opformer.training.utils import (
    train_epoch,
    validate_epoch,
    model_save,
    validate_device,
)

from mesh_opformer.layers.model import build_mesh_opformer_model


def parse_args():
    """
    Parse input arguments with `argparse`.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        type=str,
        default="./runs/",
        help="Path to directory to store model.",
    )

    parser.add_argument(
        "--cfg_path",
        type=str,
        help="Path to training configuration `.toml` file.",
    )

    args = parser.parse_args()

    return args


def parse_data_paths(folder: str) -> Dict[str, str]:
    """
    Parse paths to mesh and segmentation files of the simplified
    `human-segmentation` dataset from PD-Meshnet (Milano et al., 2020).

    Downloaded from:
        'https://www.dropbox.com/s/byk8oisbm75g5yb/human_seg.zip'

    """
    paths = {}

    for item in os.scandir(folder):

        if item.is_dir():
            continue

        name = _get_filename_from_path(item.name)

        paths[name] = item.path

    return paths


def _get_filename_from_path(path: str) -> str:
    """
    Get the number from the filename.
    """
    # Get path tail part and remove `.off` or `obj` suffix
    _, filename = os.path.split(path)
    filename, _ = os.path.splitext(filename)

    return filename


def create_split(data: Dict, labels: Dict) -> Tuple[List, List]:
    """
    Align dataset paths and labels.
    """
    split_paths, split_lbls = [], []

    for name, path in data.items():

        split_paths.append(path)
        split_lbls.append(labels[name])

    return split_paths, split_lbls


def initialize_map_datasets(cfg):
    """
    Initialize Map Dataset.
    """
    task = cfg["dataset"]["task"]

    num_classes = cfg["dataset"]["train"]["kw"]["num_classes"]

    lbls_path = cfg["dataset"]["labels_path"]

    train_set_path = cfg["dataset"]["train"]["path"]
    test_set_path = cfg["dataset"]["test"]["path"]

    lbls = parse_data_paths(lbls_path)

    train_set = parse_data_paths(train_set_path)
    test_set = parse_data_paths(test_set_path)

    # align lbls and paths
    x_train, y_train = create_split(train_set, lbls)
    x_test, y_test = create_split(test_set, lbls)

    weight = calculate_class_weights(y_train, num_classes=num_classes)

    data = {
        "train": MeshMapDataset(
            x_train, labels=y_train, task=task, **cfg["dataset"]["train"]["kw"]
        ),
        "test": MeshMapDataset(
            x_test, labels=y_test, task=task, **cfg["dataset"]["test"]["kw"]
        ),
    }

    loader = {
        name: DataLoader(
            dataset,
            **cfg["dataset"]["dataloader"][name],
            collate_fn=collate_batch_w_sparse_fn
        )
        for name, dataset in data.items()
    }

    return data, loader, weight


def calculate_class_weights(y_train: List[str], num_classes: int) -> torch.Tensor:
    """
    Determine weights to scale loss for class imbalance.
    """
    all_counts = [
        np.bincount(
            parse_segmentation_human_simplified_label(path), minlength=num_classes
        )
        for path in y_train
    ]

    counts = np.vstack(all_counts).sum(axis=0)

    weight = 1 / (counts / counts.sum())

    return torch.FloatTensor(weight)


def init_loss(loss_type="cross-entropy", **loss_kw):
    """
    Initialize loss class instance.
    """

    LOSS_CLS = {
        "cross-entropy": torch.nn.CrossEntropyLoss,
        "neg-log-likehood": torch.nn.NLLLoss,
    }

    loss_cls = LOSS_CLS.get(loss_type)

    if loss_cls is None:
        raise ValueError(
            ("Given `loss_type` argument is not available. " "Choose from {}").format(
                tuple(LOSS_CLS.keys())
            )
        )

    return loss_cls(**loss_kw)


def main():
    """
    Loads dataset, dataloaders, trains, validates and saves weights.
    """
    args = parse_args()

    with open(args.cfg_path, "r") as f:
        cfg = toml.load(f)

    # Configure device
    device = validate_device(cfg["device"])

    if device.type == "cpu":
        torch.set_num_interop_threads(4)
        torch.set_num_threads(4)

    print(device)

    # Initialize torch `Dataset` and `DataLoader` instances
    data, loader, weight = initialize_map_datasets(cfg)

    weight = weight.to(device.type)

    model = build_mesh_opformer_model(**cfg["train"]["model"]).to(device.type)

    loss_compute = init_loss(**cfg["train"]["loss"], weight=weight)

    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"], eta_min=1e-6
    )

    # Use `wandb` logger if specified
    if cfg["use_wandb"] is True:

        os.environ["WANDB_MODE"] = cfg["wandb"]["WANDB_MODE"]

        wandb.init(**cfg["wandb"]["init"], config=cfg)
        wandb.run.name = cfg["wandb"]["name"]
        wandb.watch(model, log="all", log_freq=50)
        print(wandb.run.id)

    # make model savepath if not there
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    training_kw = {
        "epochs": cfg["train"]["epochs"],
        "task": cfg["train"]["model"]["task_type"],
    }
    training_kw.update(cfg["train"]["optim"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model num. of parameters: {}".format(n_params))

    # train, val and model save
    for epoch in range(1, cfg["train"]["epochs"] + 1):

        _ = train_epoch(
            data_loader=loader["train"],
            model=model,
            loss_compute=loss_compute,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            device=device,
            training_kw=training_kw,
        )

        _ = validate_epoch(
            data_loader=loader["test"],
            model=model,
            loss_compute=loss_compute,
            epoch=epoch,
            device=device,
            training_kw=training_kw,
        )

    save_path = os.path.join(args.out, "hodgeformer_human_simplified-id_{}-lr_{}.pth".format(str(wandb.run.id), str(cfg["train"]["lr"])))

    model_save(model, optimizer, scheduler, epoch, save_path)


if __name__ == "__main__":
    main()
