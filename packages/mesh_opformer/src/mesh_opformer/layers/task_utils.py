from typing import Dict

import os
import numpy as np

from scipy import spatial


def convert_functional_map_to_pointwise_map(C12, B1, B2):
    """
    Convert a functional map to a pointwise map.

    Parameters
    ----------
    C12 : np.ndarray
        An array representing the functional map C12: S1 -> S2
    B1 : np.ndarray
        The basis of S1
    B2 : np.ndarray
        The basis of S2

    Returns
    -------
    T21 : np.ndarray
        The pointwise map T21: S2 -> S1 (index 0-based)
    """
    if C12.shape[0] != B2.shape[1] or C12.shape[1] != B1.shape[1]:
        return -1

    _, T21 = spatial.cKDTree(np.matmul(B1, C12.transpose())).query(B2)

    return T21


def convert_pointwise_map_to_functional_map(T12, B1, B2):
    """
    Convert a pointwise map to a functional map.

    Parameters
    ----------
    T12 : np.ndarray
        An array representing a pointwise map T12: S1 -> S2
    B1 : np.ndarray
        The basis of S1
    B2 : np.ndarray
        The basis of S2

    Returns
    -------
    C21 : np.ndarray
        The corresponding functional map C21: S2 -> S1
    """
    C21 = np.linalg.lstsq(B1, B2[T12, :], rcond=None)[0]

    return C21


def prebuild_eigen_cache(mesh_paths: Dict[str, str], cache_dir: str) -> None:
    """
    Precompute eigen-decomposition and store in cache.

    Parameters
    ----------
    mesh_paths : Dict[str, str]
        A dictionary of mesh names to mesh paths.
    cache_dir : str
        A path to a dicitonary to store eigencomputation results

    Returns
    -------
    None
        Stores results to `cache_dir`
    """
    import potpourri3d as pp3d

    from mesh_sim.mesh import MeshSignedIncidenceMatrices
    from mesh_sim.utils.open3d import build_o3d_mesh
    from mesh_sim.mesh_base.operators import compute_eigen

    from mesh_opformer.dataset.transforms_o3d import apply_mesh_transforms

    rs = np.random.RandomState(seed=123)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    for name, path in mesh_paths.items():

        vertices, faces = pp3d.read_mesh(path)

        _mesh_o3d = build_o3d_mesh(vertices=vertices, faces=faces)

        # Apply transformations
        transforms = ["vertex_center_scale_area"]

        apply_mesh_transforms(_mesh_o3d, transforms, rs=rs)

        mesh = MeshSignedIncidenceMatrices.from_mesh_o3d(mesh_o3d=_mesh_o3d, uid=path)

        evals, evecs = compute_eigen(
            mesh.v_s, mesh.v2e__csc, mesh.e2f__csc, mesh.v2f__csc, k=128
        )

        evecs_path = os.path.join(cache_dir, name + ".evecs.npy")
        evals_path = os.path.join(cache_dir, name + ".evals.npy")

        np.save(evecs_path, evecs)
        np.save(evals_path, evals)


class AllPairsGeodesicEngine(object):
    """
    Class to estimate all pairs geodesics.
    """

    def __init__(self, verts, faces):
        self.verts = verts
        self.faces = faces

    def __call__(self, i):
        return all_pairs_geodesic_worker(self.verts, self.faces, i)


def all_pairs_geodesic_worker(verts, faces, i):
    """
    Estimate geodesic distances between all vertices.

    TODO: this re-does a ton of work, since it is called independently each time.
    Some custom C++ code could surely make it faster.
    """
    # need libigl for geodesic call
    try:
        import igl
    except ImportError as e:
        raise ImportError(
            "Must have python libigl installed for all-pairs geodesics. `pip install libigl`"
        )

    n = verts.shape[0]

    sources = np.array([i])
    targets = np.arange(n)
    dist_vec = igl.exact_geodesic(
        verts,
        faces,
        sources,
        np.array([]),
        targets,
        np.array([]),
    )

    return dist_vec


def prebuild_geo_cache(
    mesh_paths: Dict[str, str], cache_dir: str, n_process: int = 12
) -> None:
    """
    Precompute geodesic distances and store in cache.

    Parameters
    ----------
    mesh_paths : Dict[str, str]
        A dictionary of mesh names to mesh paths.
    cache_dir : str
        A path to a dicitonary to store eigencomputation results

    Returns
    -------
    None
        Stores results to `cache_dir`
    """
    import potpourri3d as pp3d
    import multiprocessing as mp

    from mesh_sim.mesh import MeshSignedIncidenceMatrices
    from mesh_sim.utils.open3d import build_o3d_mesh
    from mesh_opformer.dataset.transforms_o3d import apply_mesh_transforms
    from mesh_opformer.layers.task_utils import AllPairsGeodesicEngine

    rs = np.random.RandomState(seed=123)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    for name, path in mesh_paths.items():

        vertices, faces = pp3d.read_mesh(path)

        _mesh_o3d = build_o3d_mesh(vertices=vertices, faces=faces)

        # Apply transformations
        transforms = ["vertex_center_scale_area"]

        apply_mesh_transforms(_mesh_o3d, transforms, rs=rs)

        mesh = MeshSignedIncidenceMatrices.from_mesh_o3d(mesh_o3d=_mesh_o3d, uid=path)

        verts = mesh.v_s

        try:
            pool = mp.Pool(n_process)
            engine = AllPairsGeodesicEngine(verts=verts, faces=faces)
            outputs = pool.map(engine, range(mesh.n_v))
        finally:  # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()

        result_dists = np.array(outputs)

        # replace any failed values with nan
        result_dists = np.nan_to_num(
            result_dists, nan=np.nan, posinf=np.nan, neginf=np.nan
        )

        # we expect that this should be a symmetric matrix, but it might not be. Take the min of the symmetric values to make it symmetric
        result_dists = np.fmin(result_dists, np.transpose(result_dists))

        # on rare occaisions MMP fails, yielding nan/inf; set it to the largest non-failed value if so
        max_dist = np.nanmax(result_dists)

        geo_dists = np.nan_to_num(
            result_dists, nan=max_dist, posinf=max_dist, neginf=max_dist
        )

        geodist_path = os.path.join(cache_dir, name + ".geodist.npz")

        np.savez(geodist_path, verts=verts, faces=faces, dist=geo_dists)


def geodesic_label_errors(
    G_s: np.ndarray,
    vts_pred: np.ndarray,
    vts_true: np.ndarray,
    area: None | float = None,
    normalization: str = "area",
):
    """
    Return a vector of distances between predicted and ground-truth lables.

    Parameters
    ----------
    G_s : np.ndarray
        A matrix with all-to-all geodesic distances.
    vts_pred : np.ndarray
        An array with predicted vertex labels.
    vts_true : np.ndarray
        An array with true vertex labels.
    area : float | None
        The total mesh area to be used for normalization. It is required
        if `normalization=area`.

    Returns
    -------
    normalized_dists : np.ndarray
        Geodesic distance errors between predicted and true labels.
    """
    _norm = ("diameter", "area")

    if normalization not in _norm:
        raise ValueError("Not valid `normalization` arg. Choose from {}".format(_norm))

    dists = G_s[vts_pred, vts_true]

    # Normalize via
    if normalization == "diameter":
        normalized_dists = dists / np.max(G_s)

    elif normalization == "area":
        normalized_dists = dists / np.sqrt(area)

    return normalized_dists
