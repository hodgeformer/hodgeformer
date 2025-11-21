import os
import toml
import argparse

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/",
        help="Path to directory where data reside.",
    )

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


def initialize_map_datasets(cfg):
    """
    Initialize Map Dataset.
    """
    if cfg["dataset"]["format"] == "json":

        data_path = cfg["dataset"]["path"]

        data = {
            "train": MeshMapDataset.from_json(
                cfg["dataset"]["json"]["train"], data_path
            ),
            "valid": MeshMapDataset.from_json(
                cfg["dataset"]["json"]["valid"], data_path
            ),
        }

    elif cfg["dataset"]["format"] == "dir":

        data = {
            "train": MeshMapDataset.from_dir(cfg["dataset"]["dir"]["train"]),
            "valid": MeshMapDataset.from_dir(cfg["dataset"]["dir"]["valid"]),
        }

    loader = {
        name: DataLoader(
            dataset,
            **cfg["dataset"]["dataloader"],
            collate_fn=collate_batch_w_sparse_fn
        )
        for name, dataset in data.items()
    }

    return data, loader


def main():
    """
    Loads dataset, dataloaders, trains, validates and saves weights.
    """
    args = parse_args()

    with open(args.cfg_path, "r") as f:
        cfg = toml.load(f)

    # Initialize transformation functions
    # ...currently None

    # Initialize torch `Dataset` and `DataLoader` instances
    data, loader = initialize_map_datasets(cfg)

    # Configure device
    device = validate_device(cfg["device"])

    if device.type == "cpu":
        torch.set_num_interop_threads(4)
        torch.set_num_threads(4)

    print(device)

    model = build_mesh_opformer_model(**cfg["train"]["model"]).to(device.type)

    loss_compute = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"], eta_min=1e-6
    )

    # Use `wandb` logger if specified
    if cfg["use_wandb"] is True:

        os.environ["WANDB_MODE"] = cfg["wandb"]["WANDB_MODE"]

        wandb.init(**cfg["wandb"]["init"])
        wandb.run.name = cfg["wandb"]["name"]
        wandb.watch(model)

    # make model savepath if not there
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    kw = {
        "batch_size": cfg["dataset"]["dataloader"]["batch_size"],
        "epochs": cfg["train"]["epochs"],
    }

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
            training_kw=kw,
        )

        _ = validate_epoch(
            data_loader=loader["valid"],
            model=model,
            epoch=epoch,
            device=device,
            training_kw=kw,
        )

        save_path = os.path.join(args.out, "mesh_opformer_ckpt_{}.pth".format(epoch))

        model_save(model, optimizer, scheduler, epoch, save_path)


if __name__ == "__main__":
    main()
