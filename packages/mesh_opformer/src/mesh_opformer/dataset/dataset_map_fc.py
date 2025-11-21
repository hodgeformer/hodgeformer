from typing import List, Dict, Tuple

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


def get_ground_truth_fmap(
    vts_s: np.ndarray,
    vts_t: np.ndarray,
    evecs_s: np.ndarray,
    evecs_t: np.ndarray,
    n_fmap: int = 30,
) -> np.ndarray:
    """
    Get correspondence ground truth functional map.
    """
    evecs_s = evecs_s[:, :n_fmap]
    evecs_t = evecs_t[:, :n_fmap]

    _evec_s, _evec_t = evecs_s[vts_s, :], evecs_t[vts_t, :]

    solve_out = np.linalg.lstsq(_evec_t, _evec_s, rcond=None)

    C_gt = solve_out[0]

    return C_gt.T


def _make_kwargs_hashable(kw: Dict):
    """
    Change kwargs data types to make them hashable.
    """
    for k, v in kw.copy().items():

        if isinstance(v, dict):
            kw.pop(k)
            kw[k] = _make_kwargs_hashable(v)

        elif isinstance(v, list):
            kw.pop(k)
            kw[k] = tuple(v)

    return kw


class MeshMapDatasetFunctionalCorrespondence(Dataset):
    """
    A class to represent a map-dataset of meshes for training.
    """

    def __init__(
        self,
        paths_meshs: List[Tuple[str, str]],
        paths_vts: List[Tuple[str, str]],
        paths_evecs: List[Tuple[str, str]],
        paths_geodists: List[Tuple[str, str]],
        transforms: None | List[str] = None,
        extract_kw: Dict = {"mode": "neighbors"},
        n_fmap: int = 30,
    ):
        """
        Initializes the `MeshMapDataset` class.

        Parameters
        ----------
        paths_meshs : List[Tuple[str, str]]
            A list of mesh path tuples.
        paths_vts : List[Tuple[str, str]]
            A list of paths to functional correspondence `vts` files.
        paths_evecs : List[Tuple[str, str]]
            A list of paths to eigenvectors and eigenvalues.
        paths_geodists : List[Tuple[str, str]]
            A list of paths to geodistances.
        transforms : List[str] | None
            Transformations to be applied on the input mesh data.
        num_classes: int | None
            Number of classes. if not available then infer from `labels`.
        extract_kw : Dict
            Kwargs to be passed in mesh feature extraction.
        n_fmap : int
            Number of eigenfunctions to keep for estimating fmap.
        """
        super().__init__()

        self.paths_meshs = paths_meshs
        self.paths_evecs = paths_evecs
        self.paths_geodists = paths_geodists
        self.paths_vts = paths_vts

        self.rs = np.random.RandomState(seed=123)

        self.transforms = tuple(transforms)

        self.extract_kw = _make_kwargs_hashable(extract_kw)

        self.n_fmap = n_fmap

    def __len__(self):
        return len(self.paths_meshs)

    def _extract_mesh_features_from_path(self, path):
        """
        Extract mesh features given a path to a mesh.

        Parameters
        ----------
        path : str
            A path to a mesh.

        Returns
        -------
        Tuple
            A tuple of tensors representing a mesh. Includes:
            * `x_v`, `x_e`, `x_f` : features on vertices, edges and faces
            * `d_0`, `d_1` : edge-vertex, face-edge incidence matrices
            * `v_idx`, `e_idx`, `f_idx` : neighborhoods on vertices, edges & faces
        evecs : np.ndarray
            Eigenvectors of the mesh Laplace Beltrami operator.
        evals : np.ndarray
            Eigenvalues of the mesh Laplace Beltrami operator.
        """
        vertices, faces = pp3d.read_mesh(path)

        _mesh_o3d = build_o3d_mesh(vertices=vertices, faces=faces)

        # Apply transformations
        if self.transforms:
            apply_mesh_transforms(_mesh_o3d, self.transforms, rs=self.rs)

        mesh = MeshSignedIncidenceMatrices.from_mesh_o3d(mesh_o3d=_mesh_o3d, uid=path)

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
        """
        mesh_path_s, mesh_path_t = self.paths_meshs[idx]
        evecs_path_s, evecs_path_t = self.paths_evecs[idx]
        vts_path_s, vts_path_t = self.paths_vts[idx]

        tensors_s = self._extract_mesh_features_from_path(mesh_path_s)
        tensors_t = self._extract_mesh_features_from_path(mesh_path_t)

        evecs_s = np.load(evecs_path_s.format("evecs"))[:, : self.n_fmap]
        evals_s = np.load(evecs_path_s.format("evals"))[: self.n_fmap]
        evecs_t = np.load(evecs_path_t.format("evecs"))[:, : self.n_fmap]
        evals_t = np.load(evecs_path_t.format("evals"))[: self.n_fmap]

        vts_s = np.loadtxt(vts_path_s).astype(int) - 1  # convert from 1-based indexing
        vts_t = np.loadtxt(vts_path_t).astype(int) - 1  # convert from 1-based indexing

        C_gt = torch.Tensor(
            get_ground_truth_fmap(vts_s, vts_t, evecs_s, evecs_t, n_fmap=self.n_fmap)
        )

        evecs_s = torch.Tensor(evecs_s)
        evals_s = torch.Tensor(evals_s)
        evecs_t = torch.Tensor(evecs_t)
        evals_t = torch.Tensor(evals_t)
        vts_s = torch.LongTensor(vts_s)
        vts_t = torch.LongTensor(vts_t)

        i_idx = torch.LongTensor([idx])

        return (
            tensors_s,
            tensors_t,
            C_gt,
            (evecs_s, evals_s, vts_s),
            (evecs_t, evals_t, vts_t),
            i_idx,
        )
