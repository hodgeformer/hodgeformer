from typing import List, Dict
from numpy.typing import ArrayLike

import os
import toml
import argparse

import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

try:
    import wandb
except ImportError:
    print("`wandb` was not found. `wandb` functionalities will be disabled.")


from mesh_opformer.dataset import MeshMapDataset
from mesh_opformer.dataset import collate_batch_w_sparse_fn

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


def _get_dataset_paths(root_dir):
    """
    Get paths for simplified mesh files.
    """
    x_train, y_train = [], []
    x_test, y_test = [], []

    ctg_map = {}

    for ctg_idx, ctg in enumerate(sorted(os.listdir(root_dir))):

        ctg_map[ctg_idx] = ctg

        ctg_dir = os.path.join(root_dir, ctg)

        if not os.path.isdir(ctg_dir):
            continue

        train_dir = os.path.join(ctg_dir, "train")
        test_dir = os.path.join(ctg_dir, "test")

        _x_train = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
        _x_test = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]

        x_train.extend(_x_train)
        x_test.extend(_x_test)

        y_train.extend([ctg_idx for _ in range(len(_x_train))])
        y_test.extend([ctg_idx for _ in range(len(_x_test))])

    return x_train, y_train, x_test, y_test, ctg_map


def calculate_class_weights(y_train: List[str], num_classes: int) -> torch.Tensor:
    """
    Determine weights to scale loss for class imbalance.
    """
    counts = np.bincount(y_train, minlength=num_classes)

    weight = 1 / (counts / counts.sum())

    return torch.FloatTensor(weight)


def initialize_map_datasets(cfg):
    """
    Initialize Map Dataset.
    """
    data_path = cfg["dataset"]["path"]
    task = cfg["dataset"]["task"]

    num_classes = cfg["dataset"]["train"]["kw"]["num_classes"]

    x_train, y_train, x_test, y_test, ctg_map = _get_dataset_paths(data_path)

    weight = calculate_class_weights(y_train, num_classes)

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

    # make model savepath if not there
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    training_kw = {"epochs": cfg["train"]["epochs"], "task": cfg["dataset"]["task"]}
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

    save_path = os.path.join(args.out, "mesh_opformer_ckpt_{}.pth".format(epoch))

    model_save(model, optimizer, scheduler, epoch, save_path)


if __name__ == "__main__":
    main()
