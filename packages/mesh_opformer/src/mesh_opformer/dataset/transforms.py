from typing import List
from numpy.typing import ArrayLike

import numpy as np

from numpy.random import RandomState


def normalize_positions(v_s: ArrayLike) -> ArrayLike:
    """
    Center and unit-scale positions.

    Parameters
    ----------
    v_s : np.ndarray
        The mesh vertices coordinates.

    Returns
    -------
    v_s : np.ndarray
        The transformed mesh vertices coordinates.
    """
    # center using point average
    v_s = v_s - np.mean(v_s, axis=0, keepdims=True)

    # scale using vertex norm 
    scale = np.linalg.norm(v_s, axis=1).max()
    v_s = v_s / scale

    return v_s


def random_rotation_matrix(rs: RandomState | int | None = 123) -> ArrayLike:
    """
    Generate a random 3D rotation matrix.
    """
    if rs is None:
        rs = RandomState()
    elif isinstance(rs, int):
        rs = RandomState(seed=rs)

    Q, _ = np.linalg.qr(rs.normal(size=(3, 3)))

    return Q


def random_scale_matrix(
    max_stretch: float = 0.1, rs: RandomState | int | None = 123
) -> ArrayLike:
    """
    Generate a random 3D anisotropic scaling matrix.
    """
    if rs is None:
        rs = RandomState()
    elif isinstance(rs, int):
        rs = RandomState(seed=rs)

    return np.diag(1 + (rs.rand(3) * 2 - 1) * max_stretch)


def apply_vertex_transforms(
    transforms: List[str], v_s: ArrayLike, rs: RandomState
) -> ArrayLike:
    """
    Apply a list of transformations on mesh vertices.
    """
    for transform in transforms:

        if transform == "rotate":
            T = random_rotation_matrix(rs=rs)
            v_s = v_s @ T

        elif transform == "scale":
            T = random_scale_matrix(max_stretch=0.1, rs=rs)
            v_s = v_s @ T

        elif transform == "vertex_center_scale":
            v_s = normalize_positions(v_s)

        else:
            raise ValueError("Given transform is not supported.")

    return v_s
