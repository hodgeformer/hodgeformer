from typing import List
from numpy.typing import ArrayLike

import numpy as np
import open3d as o3d

from numpy.random import RandomState


def normalize_mesh(mesh_o3d: o3d.geometry.TriangleMesh, how: str = "max") -> None:
    """
    Applies inplace center-scale transformations on a `o3d.geometry.TriangleMesh`
    instance.
    """
    # center mesh
    center = mesh_o3d.get_center()
    mesh_o3d.translate(-center)

    # scale mesh
    v_s = np.asarray(mesh_o3d.vertices)

    if how == "max":
        scale = 1.0 / np.linalg.norm(v_s, axis=1).max()

    elif how == "area":
        scale = 1.0 / (mesh_o3d.get_surface_area()) ** 0.5

    mesh_o3d.scale(scale, center=(0.0, 0.0, 0.0))


def random_rotate_mesh(
    mesh_o3d: o3d.geometry.TriangleMesh, rs: RandomState | int | None = 123
) -> None:
    """
    Randomly rotates inplace on a `o3d.geometry.TriangleMesh` instance.
    """
    if rs is None:
        rs = RandomState()
    elif isinstance(rs, int):
        rs = RandomState(seed=rs)

    R = mesh_o3d.get_rotation_matrix_from_xyz(
        (
            rs.uniform(0, 2 * np.pi),
            rs.uniform(0, 2 * np.pi),
            rs.uniform(0, 2 * np.pi),
        )
    )

    mesh_o3d.rotate(R, center=mesh_o3d.get_center())


def random_scale_mesh(
    mesh_o3d: o3d.geometry.TriangleMesh,
    max_stretch: float = 0.1,
    rs: RandomState | int | None = 123,
) -> None:
    """
    Performs random anisotropic scaling inplace on a `o3d.geometry.TriangleMesh`
    instance.
    """
    if rs is None:
        rs = RandomState()
    elif isinstance(rs, int):
        rs = RandomState(seed=rs)

    # Random scaling factor
    scale_factor = rs.uniform(1.0 - max_stretch, 1.0 + max_stretch)

    mesh_o3d.scale(scale_factor, center=mesh_o3d.get_center())


def random_jitter_mesh_vertices(
    mesh_o3d: o3d.geometry.TriangleMesh,
    max_jitter: float = 0.05,
    rs: RandomState | int | None = 123,
):
    """
    Perform random jitter on mesh vertex coordinates by sliding across edges.
    """
    if rs is None:
        rs = RandomState()
    elif isinstance(rs, int):
        rs = RandomState(seed=rs)

    mesh_o3d = mesh_o3d.compute_adjacency_list()

    v_s = np.asarray(mesh_o3d.vertices)

    adj_idxs = np.concatenate(
        [
            rs.choice(tuple(item), 1) if item else [-1]
            for item in mesh_o3d.adjacency_list
        ]
    )

    mask = (adj_idxs != -1).reshape(-1, 1)

    jitter = rs.uniform(-max_jitter, max_jitter, (len(v_s), 1))

    v_s = v_s + mask * jitter * (v_s[adj_idxs] - v_s)

    mesh_o3d.vertices = o3d.utility.Vector3dVector(v_s)


def apply_mesh_transforms(
    mesh_o3d: o3d.geometry.TriangleMesh, transforms: List[str], rs: RandomState
) -> None:
    """
    Apply a list of transformations on a `o3d.geometry.TriangleMesh` instance.
    """
    for transform in transforms:

        if transform == "rotate":
            random_rotate_mesh(mesh_o3d, rs=rs)

        elif transform == "scale":
            random_scale_mesh(mesh_o3d, max_stretch=0.1, rs=rs)

        elif transform == "jitter":
            random_jitter_mesh_vertices(mesh_o3d, max_jitter=0.1, rs=rs)

        elif transform == "vertex_center_scale":
            normalize_mesh(mesh_o3d, how="max")

        elif transform == "vertex_center_scale_area":
            normalize_mesh(mesh_o3d, how="area")

        else:
            raise ValueError("Given transform is not supported.")
