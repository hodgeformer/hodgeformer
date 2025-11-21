from numpy.typing import ArrayLike

import os
import toml
import argparse

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


def parse_shrec_categories_file(path):
    """
    Parse the `categories.txt` file of `shrec11` dataset.
    """
    recs, lbls = [], []
    label_names = {}

    with open(path) as f:
        # skip first line
        f.readline()
        n_classes, n_samples = [int(item) for item in f.readline().split()]

        for i in range(n_classes):
            # Skip empty line
            f.readline()
            # Get class_name, _, n_samples
            label_name, _, count = f.readline().strip().split()
            label_names[label_name] = i

            count = int(count)

            for _ in range(count):
                recs.append(f.readline().strip())
                lbls.append(i)

    assert len(recs) == n_samples

    return recs, lbls, label_names


def split_dataset(
    recs: ArrayLike,
    lbls: ArrayLike,
    train_size: float = 0.7,
    valid_size: float = 0.0,
    test_size: float = 0.3,
    random_state : int = 123,
):
    """
    Split dataset to train, valid and test sets.
    """
    train_size = int(len(recs) * train_size)
    valid_size = int(len(recs) * valid_size)
    test_size = len(recs) - train_size - valid_size

    x_train, x_test, y_train, y_test = train_test_split(
        recs,
        lbls,
        train_size=train_size + valid_size,
        test_size=test_size,
        stratify=lbls,
        random_state=random_state,
        shuffle=True,
    )

    # x_train, x_valid, y_train, y_valid = train_test_split(
    #     x_train,
    #     y_train,
    #     train_size=train_size,
    #     test_size=valid_size,
    #     stratify=y_train,
    #     random_state=123,
    #     shuffle=True,
    # )

    return x_train, x_test, y_train, y_test


def _get_dataset_original_paths(root_dir):
    """
    Get paths for original mesh files.
    """
    return os.listdir(root_dir)


def _get_dataset_simplified_paths(root_dir):
    """
    Get paths for simplified mesh files.
    """
    paths = []

    # Gather obj paths
    for dp, dn, filenames in os.walk(root_dir):
        for f in filenames:
            if os.path.splitext(f)[1] == ".obj":
                paths.append(os.path.join(dp, f))

    # Sort based on name
    return sorted(paths, key=_get_number_from_name)


def _get_number_from_name(path):
    """
    Get the number from the filename.
    """
    # Get path tail part and remove `.obj` suffix
    _, filename = os.path.split(path)
    filename, _ = os.path.splitext(filename)

    return int(filename[1:])


def initialize_map_datasets(cfg):
    """
    Initialize Map Dataset.
    """
    data_path = cfg["dataset"]["path"]
    meta_path = cfg["dataset"]["meta"]
    task = cfg["dataset"]["task"]

    if cfg["dataset"]["type"] == "original":
        file_paths = _get_dataset_original_paths(data_path)

    elif cfg["dataset"]["type"] == "simplified":
        file_paths = _get_dataset_simplified_paths(data_path)

    recs, lbls, label_names = parse_shrec_categories_file(meta_path)

    x_train, x_test, y_train, y_test = split_dataset(
        recs=recs, lbls=lbls, **cfg["dataset"]["split"]
    )

    train_paths = [file_paths[int(rec)] for rec in x_train]
    test_paths = [file_paths[int(rec)] for rec in x_test]

    data = {
        "train": MeshMapDataset(
            train_paths, labels=y_train, task=task, **cfg["dataset"]["train"]["kw"]
        ),
        "test": MeshMapDataset(
            test_paths, labels=y_test, task=task, **cfg["dataset"]["test"]["kw"]
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

    return data, loader


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

    loss_compute = init_loss(**cfg["train"]["loss"])

    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"], eta_min=1e-7
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

    save_path = os.path.join(args.out, "classification_shrec11_ckpt_{}.pth".format(epoch))

    model_save(model, optimizer, scheduler, epoch, save_path)


if __name__ == "__main__":
    main()
