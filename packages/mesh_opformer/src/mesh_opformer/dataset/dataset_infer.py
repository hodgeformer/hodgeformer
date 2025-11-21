from typing import List, Dict

import numpy as np
import torch
import potpourri3d as pp3d

from torch.utils.data import Dataset

from .transforms_o3d import apply_mesh_transforms

from .utils import sparse_scipy_to_torch

try:
    from mesh_sim.mesh import (  # pyright: ignore[reportMissingImports]
        MeshSignedIncidenceMatrices,
    )
    from mesh_sim.utils.open3d import build_o3d_mesh
    from mesh_sim.utils_ml import (
        extract_mesh_features,
    )  # pyright: ignore[reportMissingImports]
except ImportError:
    raise ImportError("Need to install mesh-sim library first.")


class MeshMapDatasetInfer(Dataset):
    """
    A class to represent a map-dataset of meshes for inference.
    """

    def __init__(
        self,
        paths: List[str],
        transforms: None | List[str] = None,
        num_classes: None | int = None,
        extract_kw: Dict = {"mode": "neighbors"},
    ):
        """
        Initializes the `MeshMapDataset` class.

        Parameters
        ----------
        paths : List[str]
            A list of mesh paths.
        transforms : List[str] | None
            Transformations to be applied on the input mesh data.
        num_classes: int | None
            Number of classes. if not available then infer from `labels`.
        extract_kw : Dict
            Kwargs to be passed in mesh feature extraction.
        """
        super().__init__()

        self.paths = paths

        self.rs = np.random.RandomState(seed=123)

        self.transforms = transforms
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

        vertices, faces = pp3d.read_mesh(path)

        _mesh_o3d = build_o3d_mesh(vertices=vertices, faces=faces)

        # Apply transformations
        if self.transforms:
            apply_mesh_transforms(_mesh_o3d, self.transforms, rs=self.rs)

        mesh = MeshSignedIncidenceMatrices.from_mesh_o3d(
            mesh_o3d=_mesh_o3d, uid=self.paths[idx]
        )

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

        return (x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx)
