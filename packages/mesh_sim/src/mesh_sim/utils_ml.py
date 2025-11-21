from typing import Tuple, List, Dict
from numpy.typing import ArrayLike

try:
    import graphblas as gb

    gb.init("suitesparse", blocking=True)
    gb.ss.config["nthreads"] = 1
    gb.ss.global_context["nthreads"] = 1

except ImportError:
    print("`graphblas` package is not available, cannot run method.")

import numpy as np
import math

from functools import cache

from mesh_sim.mesh import MeshSignedIncidenceMatrices

from .mesh_base.ops import (
    get_edge_coords_ordered,
    get_edge_dual_coords,
    get_edge_dual_lengths,
    calculate_vertex_normals,
    calculate_edge_normals,
    calculate_face_normals,
)

from .algos.bfs import (
    khop_nneighbors,
    bigbird_neighbors,
)
from .mesh_base.operators import compute_hks_autoscale


def _closest_pow_of_two(x):
    """
    Return the closest power of 2 to the input number.
    """
    return pow(2, round(math.log(x) / math.log(2)))


@cache
def extract_hks_features_from_mesh(mesh):
    """
    Extract HKS features from input mesh.
    """
    return compute_hks_autoscale(
        mesh.v_s, mesh.v2e__csc, mesh.e2f__csc, mesh.v2f__csc, num=16
    )


def extract_local_neighborhoods(
    mesh: MeshSignedIncidenceMatrices,
    v_max: int = 12,
    e_max: int = 12,
    f_max: int = 12,
    dilations: int = 1,
    v_k: int = 6,
    e_k: int = 6,
    f_k: int = 6,
    clip_nbors: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract local neighborhood index arrays for input mesh.
    """
    n_v, n_e, n_f = mesh.n_v, mesh.n_e, mesh.n_f

    # Clip nbors to enforce O(n**1.5) complexity
    if clip_nbors is True:
        nbor_v = min(v_max, int(n_v**0.5))
        nbor_e = min(e_max, int(n_e**0.5))
        nbor_f = min(f_max, int(n_f**0.5))
    else:
        nbor_v = v_max
        nbor_e = e_max
        nbor_f = f_max

    v_nbors = khop_nneighbors(
        mesh.v2e__csc, inc_type="v2e", n=nbor_v, k=v_k, dilations=dilations
    )
    e_nbors = khop_nneighbors(
        mesh.v2e__csc.T, inc_type="e2v", n=nbor_e, k=e_k, dilations=dilations
    )
    f_nbors = khop_nneighbors(
        mesh.e2f__csc.T, inc_type="f2e", n=nbor_f, k=f_k, dilations=dilations
    )

    # Always pad to `max` to support batching
    v_nbors = pad_neighbors(v_nbors, v_max)
    e_nbors = pad_neighbors(e_nbors, e_max)
    f_nbors = pad_neighbors(f_nbors, f_max)

    return v_nbors, e_nbors, f_nbors


def extract_bigbird_neighborhoods(
    mesh: MeshSignedIncidenceMatrices,
    v_max: int = 12,
    e_max: int = 12,
    f_max: int = 12,
    dilations: int = 1,
    v_k: int = 6,
    e_k: int = 6,
    f_k: int = 6,
    clip_nbors: bool = True,
    **kw,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract big-bird neighborhood index arrays for input mesh.
    """
    n_v, n_e, n_f = mesh.n_v, mesh.n_e, mesh.n_f

    # Clip nbors to enforce O(n**1.5) complexity
    if clip_nbors is True:
        nbor_v = min(v_max, int(n_v**0.5))
        nbor_e = min(e_max, int(n_e**0.5))
        nbor_f = min(f_max, int(n_f**0.5))
    else:
        nbor_v = v_max
        nbor_e = e_max
        nbor_f = f_max

    v_nbors = bigbird_neighbors(
        mesh.v2e__csc, inc_type="v2e", n=nbor_v, k=v_k, dilations=dilations, **kw
    )
    e_nbors = bigbird_neighbors(
        mesh.v2e__csc.T, inc_type="e2v", n=nbor_e, k=e_k, dilations=dilations, **kw
    )
    f_nbors = bigbird_neighbors(
        mesh.e2f__csc.T, inc_type="f2e", n=nbor_f, k=f_k, dilations=dilations, **kw
    )

    # Always pad to `max` to support batching
    v_nbors = pad_neighbors(v_nbors, v_max)
    e_nbors = pad_neighbors(e_nbors, e_max)
    f_nbors = pad_neighbors(f_nbors, f_max)

    return v_nbors, e_nbors, f_nbors


# Add cached variations of neighborhood extraction functions
extract_bigbird_neighborhoods_cached = cache(extract_bigbird_neighborhoods)
extract_local_neighborhoods_cached = cache(extract_local_neighborhoods)


def dummy_neighbors() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return empty neighbors.
    """
    v_nbors = np.empty((0, 16), dtype="int64")
    e_nbors = np.empty((0, 16), dtype="int64")
    f_nbors = np.empty((0, 16), dtype="int64")

    return v_nbors, e_nbors, f_nbors


def pad_neighbors(nbors, nbor_max):
    """
    Pad index array until reaching maximum size.
    """
    _, ncols = nbors.shape

    diff = nbor_max - ncols

    if diff == 0:
        return nbors
    else:
        return np.pad(nbors, ((0, 0), (0, diff)), "constant", constant_values=-1)


def get_vertex_features(
    mesh: MeshSignedIncidenceMatrices, feats: List = ["coords", "normals", "areas"]
) -> np.ndarray:
    """
    Extract vertex features,
    """
    _feats = {
        "coords": mesh.v_s,
        "normals": mesh.vertex_normals(),
        "areas": mesh.vertex_cell_area(),
    }

    return np.hstack([_feats[feat] for feat in feats])


def get_edge_features(
    mesh: MeshSignedIncidenceMatrices, feats: List = ["coords", "normals", "areas"]
) -> np.ndarray:
    """
    Extract edge features,
    """
    _feats = {
        "coords": get_edge_dual_coords(
            v_s=mesh.v_s, v2e=mesh.v2e__csc, e2f=mesh.e2f__csc
        ).reshape(-1, 12),
        "normals": mesh.edge_normals_from_vertices(),
        "areas": get_edge_dual_lengths(
            v_s=mesh.v_s, v2e=mesh.v2e__csc, e2f=mesh.e2f__csc
        ),
    }

    return np.hstack([_feats[feat] for feat in feats])


def get_face_features(
    mesh: MeshSignedIncidenceMatrices, feats: List = ["coords", "normals", "areas"]
) -> np.ndarray:
    """
    Extract face features,
    """
    _feats = {
        "coords": mesh.face_coords(ordered=True).reshape(-1, 9),
        "normals": mesh.face_normals(),
        "areas": mesh.face_area(),
    }

    return np.hstack([_feats[feat] for feat in feats])


def extract_mesh_features(
    mesh: MeshSignedIncidenceMatrices,
    mode: str = "neighbors",
    feats: List = ["coords", "normals", "areas"],
    nbor_kw: Dict = {},
    nbor_cache: bool = False,
) -> List[ArrayLike]:
    """
    Extract mesh features for machine learning operations.

    Parameters
    ----------
    mesh : MeshSignedIncidenceMatrices
        A mesh represented by a `MeshSignedIncidenceMatrices` instance.
    mode : str
        Type of features to return. Supports {'vertex', 'normals', 'neighbors'}
    """
    FEATS = ["coords", "normals", "areas"]

    if any(feat not in FEATS for feat in feats):
        raise ValueError(
            "Given feature argument is not available. Use a subset from `{}`".format(
                FEATS
            )
        )

    x_v = get_vertex_features(mesh, feats)
    x_e = get_edge_features(mesh, feats)
    x_f = get_face_features(mesh, feats)

    v2e = mesh.v2e__csc
    e2f = mesh.e2f__csc

    if nbor_cache is False:
        _extract_local = extract_local_neighborhoods
        _extract_bigbird = extract_bigbird_neighborhoods

    else:
        _extract_local = extract_local_neighborhoods_cached
        _extract_bigbird = extract_bigbird_neighborhoods_cached

    if mode == "neighbors":
        v_idx, e_idx, f_idx = _extract_local(mesh, **nbor_kw)

    elif mode == "bigbird":
        v_idx, e_idx, f_idx = _extract_bigbird(mesh, **nbor_kw)

    else:
        v_idx, e_idx, f_idx = dummy_neighbors()

    return x_v, x_e, x_f, v2e, e2f, v_idx, e_idx, f_idx
