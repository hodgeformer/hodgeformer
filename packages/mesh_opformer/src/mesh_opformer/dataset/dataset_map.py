from typing import List, Dict, Iterable

import numpy as np
import torch
import open3d as o3d
import potpourri3d as pp3d

from torch.utils.data import Dataset

from .transforms import apply_vertex_transforms
from .transforms_o3d import apply_mesh_transforms

from .utils import (
    sparse_scipy_to_torch,
    _from_dir,
    _from_json,
    _from_csv,
)

from .utils import (
    parse_classification_label,
    parse_segmentation_label,
    parse_segmentation_label_coseg,
    parse_segmentation_human_simplified_label,
    parse_segmentation_rna_label,
)

try:
    from mesh_sim.mesh import (  # pyright: ignore[reportMissingImports]
        MeshSignedIncidenceMatrices,
    )
    from mesh_sim.utils import (
        read_obj,
        read_off,
        read_ply,
    )  # pyright: ignore[reportMissingImports]
    from mesh_sim.utils.open3d import build_o3d_mesh
    from mesh_sim.utils_ml import (
        extract_mesh_features,
    )  # pyright: ignore[reportMissingImports]
except ImportError:
    raise ImportError("Need to install mesh-sim library first.")


class MeshMapDataset(Dataset):
    """
    A class to represent a map-dataset of meshes for training.
    """

    def __init__(
        self,
        paths: List[str],
        labels: List[str] | List[int],
        weights: None | Iterable[float] = None,
        transforms: None | List[str] = None,
        task: str = "classification",
        num_classes: None | int = None,
        extract_kw: Dict = {"mode": "neighbors"},
    ):
        """
        Initializes the `MeshMapDataset` class.

        Parameters
        ----------
        paths : List[str]
            A list of mesh paths.
        labels : List[str]
            A list of mesh labels.
        weights : Iterable[float]
            Class weights to take into consideration during training.
        transforms : List[str] | None
            Transformations to be applied on the input mesh data.
        task : str
            Based on the `task` different input data parsing.
        num_classes: int | None
            Number of classes. if not available then infer from `labels`.
        extract_kw : Dict
            Kwargs to be passed in mesh feature extraction.
        """
        super().__init__()

        self.paths = paths
        self.labels = labels

        self.rs = np.random.RandomState(seed=123)

        self.transforms = transforms

        if task == "classification":
            self.label_parser = parse_classification_label

        elif task == "segmentation":
            self.label_parser = parse_segmentation_label

        elif task == "segmentation-coseg":
            self.label_parser = parse_segmentation_label_coseg

        elif task == "segmentation-human-simplified":
            self.label_parser = parse_segmentation_human_simplified_label

        elif task == "segmentation-rna":
            self.label_parser = parse_segmentation_rna_label

        if num_classes is None:
            self.num_classes = max(self.labels)
        else:
            self.num_classes = num_classes

        if weights is None:
            self.weights = torch.FloatTensor([1])
        else:
            self.weights = torch.FloatTensor(weights)

        self.extract_kw = extract_kw

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Used by the dataloader to form batches and parallelize fetching.

        Parameters
        ----------
        idx : int
            Indicates a sample of the dataset

        Returns
        -------
        Tuple
            A tuple of tensors representing a mesh. Includes:
            * `x_v`, `x_e`, `x_f` : features on vertices, edges and faces
            * `d_0`, `d_1` : edge-vertex, face-edge incidence matrices
            * `v_idx`, `e_idx`, `f_idx` : neighborhoods on vertices, edges & faces
        label : torch.tensor
            Output classification label.
        weights : torch.tensor
            Global weights computed once at initialization to scale the loss.
            (Deprecated as they are calculated outside the Dataset class)
        """
        path = self.paths[idx]

        # if path.endswith(".obj"):
        #     loader = read_obj

        # elif path.endswith(".off"):
        #     loader = read_off

        # elif path.endswith(".ply"):
        #     loader = read_ply

        vertices, faces = pp3d.read_mesh(path)

        _mesh_o3d = build_o3d_mesh(vertices=vertices, faces=faces)

        # Apply transformations
        if self.transforms:
            apply_mesh_transforms(_mesh_o3d, self.transforms, rs=self.rs)

        mesh = MeshSignedIncidenceMatrices.from_mesh_o3d(
            mesh_o3d=_mesh_o3d, uid=self.paths[idx]
        )

        label = self.label_parser(self.labels[idx])

        x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx = extract_mesh_features(
            mesh, **self.extract_kw
        )

        x_v = torch.FloatTensor(x_v)
        x_e = torch.FloatTensor(x_e)
        x_f = torch.FloatTensor(x_f)
        d_0 = sparse_scipy_to_torch(d_0, transpose=True)
        d_1 = sparse_scipy_to_torch(d_1, transpose=True)

        v_idx = torch.LongTensor(v_idx)
        e_idx = torch.LongTensor(e_idx)
        f_idx = torch.LongTensor(f_idx)

        return (
            (x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx),
            label,
            self.weights,
        )

    @classmethod
    def from_json(cls, json_path, data_dir):
        """
        Initialize `MeshMapDataset` instance from `.json` file.
        """
        paths, labels = _from_json(json_path, data_dir)

        return cls(paths=paths, labels=labels)

    @classmethod
    def from_dir(cls, data_dir):
        """
        Initialize `MeshMapDataset` instance from path to rood diretory.
        """
        paths, labels = _from_dir(data_dir)

        return cls(paths=paths, labels=labels)

    @classmethod
    def from_csv(cls, csv_path, data_dir):
        """
        Initialize `MeshMapDataset` instance from `.csv` file.
        """
        paths, labels = _from_csv(csv_path, data_dir)

        return cls(paths=paths, labels=labels)
