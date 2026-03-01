from typing import List, Dict, Tuple

import os
import toml
import json
import argparse

import torch

from torch import nn
from torch.utils.data import DataLoader

from mesh_opformer.dataset import MeshMapDatasetInfer
from mesh_opformer.dataset import collate_batch_w_sparse_fn_infer

from mesh_opformer.training.utils import (
    infer_epoch,
    validate_device,
)

from mesh_opformer.layers.model import build_mesh_opformer_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg_path",
        type=str,
        help="Path to training configuration `.toml` file.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the `.pth` trained model.",
    )

    parser.add_argument(
        "--out",
        type=str,
        help="Path where the results will be saved.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data_path",
        action="store",
        dest="data_path",
        help="Path to instance to generate predictions for.",
    )

    group.add_argument(
        "--dataset_path",
        action="store",
        dest="dataset_path",
        help="Path to dataset with instances to generate predictions for.",
    )

    args = parser.parse_args()

    return args


def parse_dataset(dataset_path: str) -> List[str]:
    """
    Load dataset.
    """
    paths = [os.path.join(dataset_path, p) for p in os.listdir(dataset_path)]

    return paths


def initialize_map_datasets(paths: List[str], cfg: Dict) -> Tuple[Dict, Dict]:
    """
    Initialize Map Dataset.
    """
    dataset = MeshMapDatasetInfer(paths, **cfg["dataset"]["kw"])

    loader = DataLoader(
        dataset,
        **cfg["dataset"]["dataloader"],
        collate_fn=collate_batch_w_sparse_fn_infer
    )

    return dataset, loader


def load_model(model_path: str, cfg_model: Dict, device: torch.device) -> nn.Module:
    """
    Load model from state_dict.
    """
    model = build_mesh_opformer_model(**cfg_model).to(device.type)

    # Check `model_save` from `mesh_opformer.training.utils`
    state_dict = torch.load(model_path, map_location=device)["state"]

    try:
        missing_keys, unexpected_fields = model.load_state_dict(state_dict)

        print(missing_keys, unexpected_fields)
    except Exception as e:
        print("An error has occured when loading the model: {}".format(e))

    return model


def main():
    """
    Loads dataset, dataloaders, trains, validates and saves weights.
    """
    args = parse_args()

    with open(args.cfg_path, "r") as f:
        cfg = toml.load(f)

    ## Configure device
    device = validate_device(cfg["device"])

    if device.type == "cpu":
        torch.set_num_interop_threads(4)
        torch.set_num_threads(4)

    print(device)

    ## Prepare inference data
    if args.data_path:
        paths = [args.data_path]

    elif args.dataset_path:
        paths = parse_dataset(args.dataset_path)

    _, data_loader = initialize_map_datasets(paths, cfg)

    ## Prepare model
    model = load_model(args.model_path, cfg["model"], device)

    # Perform inference
    kw = {"task": cfg["model"]["task_type"]}

    preds = infer_epoch(data_loader=data_loader, model=model, device=device, kw=kw)

    results = {path: pred for path, pred in zip(paths, preds)}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
