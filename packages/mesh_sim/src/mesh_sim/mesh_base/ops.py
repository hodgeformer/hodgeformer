from numpy.typing import ArrayLike

import itertools

import numpy as np

from scipy.sparse import csc_matrix, coo_matrix

# ops with `csc_matrix`
from .maps import (
    face_to_edge,
    face_to_vtx,
    edge_to_vtx,
)

# ops with `csr_matrix`
from .maps import (
    edge_to_face,
    vtx_to_face,
    vtx_to_edge,
)


def get_edge_vectors(
    v_s: ArrayLike, v2e: csc_matrix, e_idx: ArrayLike | slice | None = None
) -> ArrayLike:
    """
    Get edge vectors.
    """
    return edge_to_vtx(v2e, e_idx=e_idx, return_sparse=True).T.dot(v_s)


def get_unit_vectors(vecs, inplace=False):
    """
    get unit vectors.
    """
    length = np.linalg.norm(vecs, axis=-1, keepdims=True)

    if inplace is True:
        raise NotImplementedError

    out = vecs if inplace is True else np.zeros_like(vecs)

    out = np.divide(vecs, length, out=out, where=length > 1e-5)

    return out


def get_edge_coords_unordered(
    v_s: ArrayLike, v2e: csc_matrix, e_idx: ArrayLike | slice | None = None
):
    """
    Get edge coordinates ordered by the edge orientation.
    """
    v_idx__e, _ = edge_to_vtx(v2e, e_idx=e_idx, return_sparse=False)

    return v_s[v_idx__e]


def get_edge_coords_ordered(
    v_s: ArrayLike, v2e: csc_matrix, e_idx: ArrayLike | slice | None = None
):
    """
    Get edge coordinates ordered by the edge orientation.
    """
    v_idx__e = get_edge_indices_ordered(v2e, e_idx=e_idx)

    return v_s[v_idx__e]


def get_face_coords_unordered(
    v_s: ArrayLike,
    v2f: csc_matrix,
    f_idx: ArrayLike | slice | None = None,
):
    """
    Get face coordinates unordered.
    """
    v_idx__f = face_to_vtx(v2f=v2f, f_idx=f_idx, return_sparse=False)

    return v_s[v_idx__f]


def get_face_coords_ordered(
    v_s: ArrayLike,
    v2e: csc_matrix,
    e2f: csc_matrix,
    f_idx: ArrayLike | slice | None = None,
):
    """
    Get face coordinates ordered by the face orientation.
    """
    v_idx__f = get_face_indices_ordered(v2e=v2e, e2f=e2f, f_idx=f_idx)

    return v_s[v_idx__f]


def get_edge_indices_ordered(v2e: csc_matrix, e_idx: ArrayLike | slice | None = None):
    """
    For each edge return edge indices ordered based on edge orientation.

    Parameters
    ----------
    v2e : scipy.sparse.csc_matrix
        A vertex-to-edge incidence matrix in `csc` format.
    e_idx : int, slice, np.ndarray
        Edge index or indices.

    Returns
    -------
    np.ndarray
        An array of shape `(n_e, 2)` with the edge indices ordered based
        on edge orientation, where `n_e` the number of edges.
    """
    v_idx, v_sgn = edge_to_vtx(v2e, e_idx=e_idx, return_sparse=False)

    v_sgn[v_sgn == -1] = 0

    return np.take_along_axis(v_idx, v_sgn, axis=1)


def get_face_indices_ordered(
    v2e: csc_matrix, e2f: csc_matrix, f_idx: ArrayLike | slice | None = None
):
    """
    For each face return face indices ordered based on face orientation.

    Parameters
    ----------
    v2e : scipy.sparse.csc_matrix
        A vertex-to-edge incidence matrix in `csc` format.
    e2f : scipy.sparse.csr_matrix
        An edge-to-face incidence matrix in `csr` format.
    f_idx : int, slice, np.ndarray
        Face index or indices.

    Returns
    -------
    np.ndarray
        An array of shape `(n_f, 3)` with the face indices ordered based
        on face orientation, where `n_f` the number of faces.
    """
    # Take edge indices for given faces
    e_idx, e_sgn = face_to_edge(e2f=e2f, f_idx=f_idx)

    # Pick first and second edges
    e1, e2 = 0, 1

    v1_idx, v1_sgns = edge_to_vtx(v2e, e_idx=e_idx[:, e1], return_sparse=False)
    v2_idx, v2_sgns = edge_to_vtx(v2e, e_idx=e_idx[:, e2], return_sparse=False)

    # Filp with edge signs
    v1_sgns = v1_sgns * e_sgn[:, [e1]]
    v2_sgns = v2_sgns * e_sgn[:, [e2]]

    # Order vertex indices based on edge orientation
    v1_sgns[v1_sgns == -1] = 0
    v2_sgns[v2_sgns == -1] = 0

    v1_idx = np.take_along_axis(v1_idx, v1_sgns, axis=1)
    v2_idx = np.take_along_axis(v2_idx, v2_sgns, axis=1)

    other_idx = _get_v_idx_diff(v1_idx, v2_idx)

    v_f_idx = np.column_stack([v1_idx, other_idx])

    return v_f_idx


def _get_v_idx_diff(v1_idx, v2_idx):
    """
    Compare two vertex index arrays
    """
    # Get mask of indices existing in both arrays
    mask = np.bitwise_or(v1_idx == v2_idx, v1_idx[:, ::-1] == v2_idx)

    return v2_idx[np.bitwise_not(mask)]


def get_vertex_dual_indices(v2e: csc_matrix, v_idx):
    """ """
    pass


def get_vertex_indices_from_dual_edges(
    v2e: csc_matrix,
    e2f: csc_matrix,
    f_idx: ArrayLike | slice | None = None,
) -> ArrayLike:
    """
    Get edge vertices and edge-opposite vertices indices.
    """
    if f_idx is not None:
        raise NotImplementedError("Not supported yet for specific `f_idx`")

    # Take edge indices for given faces
    e_idx, _ = face_to_edge(e2f=e2f, f_idx=None)

    # Pick first and second edges e1, e2, e3.
    e1, e2, e3 = 0, 1, 2

    # Get vertices for edges e1, e2, e3
    v_idxs__e1, _ = edge_to_vtx(v2e, e_idx=e_idx[:, e1], return_sparse=False)
    v_idxs__e2, _ = edge_to_vtx(v2e, e_idx=e_idx[:, e2], return_sparse=False)
    v_idxs__e3, _ = edge_to_vtx(v2e, e_idx=e_idx[:, e3], return_sparse=False)

    # For each edge in face find the opposite vertex
    opp_idx__e1 = _get_v_idx_diff(v_idxs__e1, v_idxs__e2)
    opp_idx__e2 = _get_v_idx_diff(v_idxs__e2, v_idxs__e3)
    opp_idx__e3 = _get_v_idx_diff(v_idxs__e3, v_idxs__e1)

    # Organize opposite vertices per edge
    # Given e_idx & (v_idxs__e1, opp_idx__e1), (v_idxs__e2, opp_idx__e2), (v_idxs__e3, opp_idx__e3)
    # For each edge gather ordered indices & dual indices (cant avoid for loops -> cython?)
    #
    # To avoid for loops:
    # (1) build coo matrix
    # (2) transform to csr
    # (3) slice data entries with indptr
    n_e, n_f = e2f.shape

    r_idx = e_idx.reshape(-1)
    c_idx = np.repeat(np.arange(n_f), 3)
    data = np.column_stack([opp_idx__e1, opp_idx__e2, opp_idx__e3]).reshape(-1)

    _coo = coo_matrix((data, (r_idx, c_idx)), shape=(n_e, n_f))
    _csr = _coo.tocsr()
    _csr.sort_indices()

    _tmp = np.split(_csr.data, _csr.indptr)[1:-1]

    e_opp_idx = np.stack(
        [
            (
                arr
                if len(arr) == 2
                else np.pad(
                    arr[:2], pad_width=(0, 2 - len(arr[:2])), constant_values=-1
                )
            )
            for arr in _tmp
        ]
    )

    # Get edge indices per edge ordered by orientation
    e_ord_idx = get_edge_indices_ordered(v2e=v2e, e_idx=None)

    # Stack dual and opposite vertex indices
    return np.column_stack([e_ord_idx, e_opp_idx])


def get_edge_dual_coords(
    v_s: ArrayLike,
    v2e: csc_matrix,
    e2f: csc_matrix,
    f_idx: ArrayLike | slice | None = None,
) -> ArrayLike:
    """
    Get edge vertices and edge-opposite vertices coordinates.
    """
    e_dual_idx = get_vertex_indices_from_dual_edges(v2e=v2e, e2f=e2f, f_idx=f_idx)

    # Take coordinates
    dual_coords = v_s[e_dual_idx]

    # Replace missing values for boundary triangles
    mask = e_dual_idx == -1
    dual_coords[mask, :] = 0.0

    return dual_coords


def get_edge_dual_lengths(
    v_s: ArrayLike,
    v2e: csc_matrix,
    e2f: csc_matrix,
    f_idx: ArrayLike | slice | None = None,
    how: str = "combination",
) -> ArrayLike:
    """
    Get edge lengths of the edge vertices and the edge-opposite vertices.

    There two different modes supported which are controlled by the `how`
    argument. The returned edge lengths are ordered based on edge orientation.

    if `v_s` and `v_t` the source, target edge vertices and `v_opp1`
    and `v_opp2` the opposite edge vertices, the resulting lengths
    are returned in the following order based on `how`:

    ```
    if how == "opposite":
        [
            (v_st, v_tg),
            (v_opp1, v_opp2),
        ]

    if how == "combination":
        [
            (v_st, v_tg),
            (v_st, v_opp1),
            (v_st, v_opp2),
            (v_tg, v_opp1),
            (v_tg, v_opp2),
            (v_opp1, v_opp2),
        ]
    ```

    Parameters
    ----------
    v_s : np.ndarray
        The mesh vertices coordinates.
    v2e : scipy.sparse.csc_matrix
        A vertex-to-edge incidence matrix in `csc` format.
    e2f : scipy.sparse.csr_matrix
        An edge-to-face incidence matrix in `csr` format.
    f_idx : int, slice, np.ndarray
        Face index or indices.
    how : {'combination', 'opposite'}
        How the edge lengths will be calculated.

    Returns
    -------
    e_dual_lens : np.ndarray
        An array of shape `(n_e, 2)` or `(n_e, 6)` based on the `how` argument
        with the edge lengths ordered based on edge orientation.
    """
    HOWS = ("opposite", "combination")

    if how not in HOWS:
        raise ValueError(
            "Given `how` argument is not valid. Choose from {}".format(HOWS)
        )

    # Get oriented edge vertices and its duals
    e_dual_idx = get_vertex_indices_from_dual_edges(v2e=v2e, e2f=e2f, f_idx=f_idx)

    # Get combinations of vertices
    if how == "opposite":
        combs = np.array([[0, 1], [2, 3]])

    elif how == "combination":
        combs = np.array(list(itertools.combinations(range(4), 2)))

    # Get vertex indices & coords based on combinations
    e_dual_combs_idx = e_dual_idx[:, combs]

    e_dual_combs_coords = v_s[e_dual_combs_idx]

    # Calcualate edge lengths
    e_dual_lens = np.linalg.norm(np.diff(e_dual_combs_coords, axis=2).squeeze(), axis=2)

    # Replace missing edges for boundary triangles with zeros
    mask = np.any(e_dual_combs_idx == -1, axis=2)

    e_dual_lens[mask] = 0.0

    return e_dual_lens


def calculate_edge_lengths(
    v_s: ArrayLike, v2e: csc_matrix, e_idx: ArrayLike | slice | None = None
) -> ArrayLike:
    """
    Calculate edge lengths.

    Parameters
    ----------
    v_s : np.ndarray
        The mesh vertices coordinates.
    v2e : scipy.sparse.csc_matrix
        A vertex-to-edge incidence matrix in `csc` format.
    e_idx : int, slice, np.ndarray
        Edge index or indices.

    Returns
    -------
    np.ndarray
        An array of shape `(n_e,)` with the length for each edge
        where `n_e`, the number of edges.
    """
    e_vecs = get_edge_vectors(v_s, v2e, e_idx)

    return np.linalg.norm(e_vecs, axis=1)


def calculate_edge_normals(
    v_s: ArrayLike,
    v2e: csc_matrix,
    e2f: csc_matrix,
    e_idx: ArrayLike | slice | None = None,
    base_normals: ArrayLike | None = None,
    base_elem: str = "v",
) -> ArrayLike:
    """
    Calcualte edge normals for edge indices.
    """
    if base_elem == "f":

        f_idx = edge_to_face(e2f=e2f, e_idx=e_idx)

        if base_normals is None:
            base_normals = calculate_face_normals(v_s, v2e, e2f, f_idx=None)

        e_normals = np.stack([base_normals[f].mean(axis=0) for f in f_idx], axis=0)

    elif base_elem == "v":

        v_idx, _ = edge_to_vtx(v2e=v2e, e_idx=e_idx)

        if base_normals is None:
            raise NotImplementedError

        e_normals = np.stack([base_normals[v].mean(axis=0) for v in v_idx], axis=0)

    return get_unit_vectors(e_normals)


def calculate_face_normals(
    v_s: ArrayLike,
    v2e: csc_matrix,
    e2f: csc_matrix,
    f_idx: ArrayLike | slice | None = None,
) -> ArrayLike:
    """
    Calculate face normals for face indices.

    Parameters
    ----------
    v_s : np.ndarray
        The mesh vertices coordinates.
    v2e : scipy.sparse.csc_matrix
        A vertex-to-edge incidence matrix in `csc` format.
    e2f : scipy.sparse.csc_matrix
        An edge-to-face incidence matrix in `csc` format.
    f_idx : int, slice, np.ndarray
        Face index or indices to calculate their normals.

    Returns
    -------
    np.ndarray
        The calculated face normals as an array of shape `(n_f, 3)`
        where `n_f` the number of faces.
    """
    v_f_idx = get_face_coords_ordered(v_s, v2e, e2f, f_idx)

    e1_vec = v_f_idx[:, 1] - v_f_idx[:, 0]
    e2_vec = v_f_idx[:, 2] - v_f_idx[:, 0]

    # Take edge indices for given faces
    # e_idx, e_sgn = face_to_edge(e2f=e2f, f_idx=f_idx)

    # # Pick first and second edges
    # e1, e2 = 0, 1

    # e1_vec = get_edge_vectors(v_s, v2e, e_idx=e_idx[:, e1])
    # e2_vec = get_edge_vectors(v_s, v2e, e_idx=e_idx[:, e2])

    # # Flip with edge sign if required
    # e1_vec = e1_vec * e_sgn[:, [e1]]
    # e2_vec = e2_vec * e_sgn[:, [e2]]

    f_normals = np.cross(e1_vec, e2_vec)

    return get_unit_vectors(f_normals)


def calculate_face_centroids(
    v_s: ArrayLike, v2f: csc_matrix, f_idx: ArrayLike | slice | None = None
) -> ArrayLike:
    """
    Caclculate face centroids for face indices.

    Parameters
    ----------
    v_s : np.ndarray
        The mesh vertices coordinates.
    v2f : scipy.sparse.csc_matrix
        A vertex-to-face incidence matrix in `csc` format.
    f_idx : int, slice, np.ndarray
        Face index or indices to calculate face centroids for.

    Returns
    -------
    np.ndarray
        The calculated face centroids as an array of shape `(n_f, 3)`
        where `n_f` the number of faces.
    """
    # Get a sparse matrix of shape `(n_v, n_f)`
    # where n_v = len(v_s), n_f = len(f_idx)
    _v2f = face_to_vtx(v2f=v2f, f_idx=f_idx, return_sparse=True)

    return _v2f.T.dot(v_s) / 3


def calculate_face_area(
    v_s: ArrayLike,
    v2e: csc_matrix,
    e2f: csc_matrix,
    f_idx: ArrayLike | slice | None = None,
) -> ArrayLike:
    """
    Caclculate face area for given face indices.

    Parameters
    ----------
    v_s : np.ndarray
        The mesh vertices coordinates.
    v2e : scipy.sparse.csc_matrix
        A vertex-to-edge incidence matrix in `csc` format.
    e2f : scipy.sparse.csc_matrix
        An edge-to-face incidence matrix in `csc` format.
    f_idx : int, slice, np.ndarray
        Face index or indices to calculate face area.

    Returns
    -------
    np.ndarray
        The calculated face area as an array of shape `(n_f,)`
        where `n_f` the number of faces.
    """
    # Take edge indices for given faces
    e_idx, _ = face_to_edge(e2f=e2f, f_idx=f_idx)

    # Pick first and second edges
    e1, e2 = 0, 1

    e1_vec = get_edge_vectors(v_s, v2e, e_idx=e_idx[:, e1])
    e2_vec = get_edge_vectors(v_s, v2e, e_idx=e_idx[:, e2])

    f_unsigned_normals = np.cross(e1_vec, e2_vec)

    return np.linalg.norm(f_unsigned_normals, axis=1, keepdims=True) / 2


def calculate_vertex_cell_area(
    v_s: ArrayLike,
    v2e: csc_matrix,
    e2f: csc_matrix,
    v2f: csc_matrix,
    v_idx: ArrayLike | slice | None = None,
) -> ArrayLike:
    """
    Calculate vertex cell area.

    This calculation is performed by summing for each vertex its adjacent faces
    and dividing by 3:

                    A_i = 1/3 * sum(i -> N(i)) area(T_i)

    Parameters
    ----------
    v_s : np.ndarray
        The mesh vertices coordinates.
    v2e : scipy.sparse.csc_matrix
        A vertex-to-edge incidence matrix in `csc` format.
    e2f : scipy.sparse.csc_matrix
        An edge-to-face incidence matrix in `csc` format.
    v2f : scipy.sparse.csc_matrix
        A vertex-to-face incidence matrix in `csc` format.
    v_idx : int, slice, np.ndarray
        Vertex index or indices to get their associating faces.

    Returns
    -------
    np.ndarray
        The calculated vertex normals as an array of shape `(n_v, 3)`
        where `n_v` the number of vertices.
    """
    f_indices = vtx_to_face(v2f, v_idx)

    # Get unique indices
    f_idx = np.unique(np.concatenate(f_indices))

    # In the case `f_idx` does not correspond to all available faces
    # we need to reindex an input to get the corresponding face normal
    # or face area.
    _indexer = {idx: i for i, idx in enumerate(f_idx)}
    _reindex = lambda f_idx: [_indexer[idx] for idx in f_idx]

    f_area = calculate_face_area(v_s, v2e, e2f, f_idx)

    return np.array(
        [(1 / 3) * np.sum(f_area[_reindex(f_idx)]) for f_idx in f_indices]
    ).reshape(-1, 1)


def calculate_vertex_normals(
    v_s: ArrayLike,
    v2e: csc_matrix,
    e2f: csc_matrix,
    v2f: csc_matrix,
    v_idx: ArrayLike | slice | None = None,
    how: str = "area_weighted",
    f_normals: ArrayLike | None = None,
) -> ArrayLike:
    """
    Calculate vertex normal vectors.

    There are several different definitions for calcualting vertex normals:

    - `uniform_weighting`:
        Average of neighboring face normals. Easy to calculate but sensitive
        to noise.

    - `volume_gradient`:
        Gradient Direction of the enclosed volume.

    - `area_weighted`:
        Weighted vector sum of the normals of incident faces, with the
        weight for each normal the area of the associated face.

    Parameters
    ----------
    v_s : np.ndarray
        The mesh vertices coordinates.
    v2e : scipy.sparse.csc_matrix
        A vertex-to-edge incidence matrix in `csc` format.
    e2f : scipy.sparse.csc_matrix
        An edge-to-face incidence matrix in `csc` format.
    v2f : scipy.sparse.csc_matrix
        A vertex-to-face incidence matrix in `csc` format.
    v_idx : int, slice, np.ndarray
        Vertex index or indices to get their associating faces.
    how : str
        Choose method for calculating vertex normals.

    Returns
    -------
    np.ndarray
        The calculated vertex normals as an array of shape `(n_v, 3)`
        where `n_v` the number of vertices.
    """
    hows = (
        "uniform_weighting",
        "volume_gradient",
        "area_weighted",
    )

    if how not in hows:
        raise ValueError(
            "Input `how` argument is not valid. Choose from {}".format(hows)
        )

    f_indices = vtx_to_face(v2f, v_idx)

    # Get unique indices
    f_idx = np.unique(np.concatenate(f_indices))

    if f_normals is None:
        f_normals = calculate_face_normals(v_s, v2e, e2f, f_idx=f_idx)

    # In the case `f_idx` does not correspond to all available faces
    # we need to reindex an input to get the corresponding face normal
    # or face area.
    _indexer = {idx: i for i, idx in enumerate(f_idx)}
    _reindex = lambda f_idx: [_indexer[idx] for idx in f_idx]

    if how == "uniform_weighting":

        v_normals = _vertex_normals__uniform_weighting(
            f_indices=f_indices, f_normals=f_normals, reindex=_reindex
        )

    elif how == "volume_gradient":

        f_area = calculate_face_area(v_s, v2e, e2f, f_idx)

        v_normals = _vertex_normals__volume_gradient(
            f_indices=f_indices, f_normals=f_normals, f_area=f_area, reindex=_reindex
        )

    elif how == "area_weighted":

        f_area = calculate_face_area(v_s, v2e, e2f, f_idx)

        v_normals = _vertex_normals__area_weighted(
            f_indices=f_indices, f_normals=f_normals, f_area=f_area, reindex=_reindex
        )

    v_normals = np.array(v_normals)

    return get_unit_vectors(v_normals)


def _vertex_normals__uniform_weighting(f_indices, f_normals, reindex):
    """
    Calculate vertex normals with the `uniform weighting` method.
    """
    return [np.mean(f_normals[reindex(f_idx)], axis=0) for f_idx in f_indices]


def _vertex_normals__volume_gradient(f_indices, f_normals, f_area, reindex):
    """
    Calculate vertex normals with the `volume gradient` method.
    """
    return [
        np.mean(
            1.0 / 3.0 * f_normals[reindex(f_idx)] * f_area[reindex(f_idx)],
            axis=0,
        )
        for f_idx in f_indices
    ]


def _vertex_normals__area_weighted(f_normals, f_indices, f_area, reindex):
    """
    Calculate vertex normals with the `area weighted` method.
    """
    return [
        np.mean(f_normals[reindex(f_idx)] * f_area[reindex(f_idx)], axis=0)
        for f_idx in f_indices
    ]


def calculate_face_edges_lengths(
    v_s: ArrayLike,
    v2e: csc_matrix,
    e2f: csc_matrix,
    f_idx: ArrayLike | slice | None = None,
) -> ArrayLike:
    """
    For each edge in a triangle calculate its length.

    Parameters
    ----------
    v_s : np.ndarray
        The mesh vertices coordinates.
    v2e : scipy.sparse.csc_matrix
        A vertex-to-edge incidence matrix in `csc` format.
    e2f : scipy.sparse.csc_matrix
        An edge-to-face incidence matrix in `csc` format.
    f_idx : int, slice, np.ndarray
        Face index or indices.

    Returns
    -------
    np.ndarray
        An array of shape `(n_f, 3)` with the edge lengths
        for each face, where `n_f` the number of faces.
    """
    # Take edge indices for given faces
    e_idx, _ = face_to_edge(e2f=e2f, f_idx=f_idx)

    # Pick first and second edges
    e1, e2, e3 = 0, 1, 2

    e1_lens = calculate_edge_lengths(v_s, v2e, e_idx=e_idx[:, e1])
    e2_lens = calculate_edge_lengths(v_s, v2e, e_idx=e_idx[:, e2])
    e3_lens = calculate_edge_lengths(v_s, v2e, e_idx=e_idx[:, e3])

    return np.stack([e1_lens, e2_lens, e3_lens], axis=1)


def calculate_edges_opposite_angles(
    v_s: ArrayLike,
    v2e: csc_matrix,
    e2f: csc_matrix,
    f_idx: ArrayLike | slice | None = None,
) -> ArrayLike:
    """
    For each edge in a triangle face calculate its opposite angle.

    Parameters
    ----------
    v_s : np.ndarray
        The mesh vertices coordinates.
    v2e : scipy.sparse.csc_matrix
        A vertex-to-edge incidence matrix in `csc` format.
    e2f : scipy.sparse.csr_matrix
        An edge-to-face incidence matrix in `csr` format.
    f_idx : int, slice, np.ndarray
        Face index or indices.

    Returns
    -------
    np.ndarray
        An array of shape `(n_f, 3)` with the edge opposite angles
        for each face, where `n_f` the number of faces.
    """
    e_dual_idx = get_vertex_indices_from_dual_edges(v2e=v2e, e2f=e2f, f_idx=f_idx)

    # Pick vertices
    pick__s_e_to_a = np.array([[0, 2], [1, 2]], dtype="int32")
    pick__s_e_to_b = np.array([[0, 3], [1, 3]], dtype="int32")

    idx__s_e_to_a = e_dual_idx[:, pick__s_e_to_a]
    idx__s_e_to_b = e_dual_idx[:, pick__s_e_to_b]

    vec__s_e_to_a = np.diff(v_s[idx__s_e_to_a], axis=2).squeeze()
    vec__s_e_to_b = np.diff(v_s[idx__s_e_to_b], axis=2).squeeze()

    vec__s_e_to_a = get_unit_vectors(vec__s_e_to_a)
    vec__s_e_to_b = get_unit_vectors(vec__s_e_to_b)

    dot_a = np.einsum("ij, ij -> i", vec__s_e_to_a[:, 0], vec__s_e_to_a[:, 1])
    dot_b = np.einsum("ij, ij -> i", vec__s_e_to_b[:, 0], vec__s_e_to_b[:, 1])

    ang_a = np.arccos(np.clip(dot_a, -1.0, 1.0))
    ang_b = np.arccos(np.clip(dot_b, -1.0, 1.0))

    ang_a[e_dual_idx[:, 2] == -1] = 0
    ang_b[e_dual_idx[:, 3] == -1] = 0

    # Calculated angles are the exterior angles. The interior angle
    # is the supplement angle, so we just subtract from `np.pi`
    return np.column_stack([ang_a, ang_b])
