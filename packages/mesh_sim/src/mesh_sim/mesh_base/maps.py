from typing import Tuple, List
from numpy.typing import ArrayLike

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix


def face_to_edge(
    e2f: csc_matrix, f_idx: ArrayLike | slice | None = None, return_sparse: bool = False
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Get edge indices from given face indices.

    Parameters
    ----------
    e2f : scipy.sparse.csc_matrix
        An edge-to-face incidence matrix in `csc` format.
    f_idx : int, slice, np.ndarray
        Face index or indices to get edge indices for.
    return_sparse : bool
        Whether to return a sparse matrix.

    Returns
    -------
    e_idx : np.ndarray
        An array of shape `(n_f, 3)` with the edge indices
        for each face, where `n_f` the number of faces.
    e_sgn : np.ndarray
        An array of shape `(n_f, 3)` with the edge signs
        for each face, where `n_f` the number of faces.
    """
    if not isinstance(e2f, csc_matrix):
        e2f = e2f.tocsc()

    if f_idx is None:
        _e2f = e2f

    else:
        if not isinstance(f_idx, (int, slice, list, np.ndarray)):
            raise ValueError(
                "`f_idx` must be of type `int`, `slice` or an index array."
            )

        _e2f = e2f[:, f_idx]

    if return_sparse is True:
        return _e2f

    e_idx = np.vstack(np.split(_e2f.indices, _e2f.indptr)[1:-1])
    e_sgn = np.vstack(np.split(_e2f.data, _e2f.indptr)[1:-1])

    return e_idx, e_sgn


def edge_to_face(
    e2f: csr_matrix, e_idx: ArrayLike | slice | None = None, return_sparse: bool = False
) -> List[ArrayLike]:
    """
    Get face indices from given edge indices.

    Parameters
    ----------
    e2f : scipy.sparse.csr_matrix
        An edge-to-face incidence matrix in `csr` format.
    e_idx : int, slice, np.ndarray
        Edge index or indices to get their face indices.
    return_sparse : bool
        Whether to return a sparse matrix.

    Returns
    -------
    f_idx : List[np.ndarray]
        A list with arrays with the face indices for each
        edge.
    """
    if not isinstance(e2f, csr_matrix):
        e2f = e2f.tocsr()

    if e_idx is None:
        _e2f = e2f

    else:
        if not isinstance(e_idx, (int, slice, list, np.ndarray)):
            raise ValueError(
                "`f_idx` must be of type `int`, `slice` or an index array."
            )

        _e2f = e2f[e_idx, :]

    if return_sparse is True:
        return _e2f

    return np.split(_e2f.indices, _e2f.indptr)[1:-1]


def edge_to_vtx(
    v2e: csc_matrix, e_idx: ArrayLike | slice | None = None, return_sparse: bool = False
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Get vertex indices from given edge indices.

    Parameters
    ----------
    v2e : scipy.sparse.csc_matrix
        A vertex-to-edge incidence matrix in `csc` format.
    e_idx : int, slice, list, np.ndarray or None
        Edge index or indices to get vertex indices for.
    return_sparse : bool
        Whether to return a sparse matrix.

    Returns
    -------
    v_idx : np.ndarray
        An array of shape `(n_e, 2)` with the vertex indices
        of each edge, where `n_e` the number of edges.
    v_sgn : np.ndarray
        An array of shape `(n_e, 2)` with the vertex signs
        of each edge, where `n_e` the number of edges.
    """
    if not isinstance(v2e, csc_matrix):
        v2e = v2e.tocsc()

    if e_idx is None:
        _v2e = v2e

    else:
        if not isinstance(e_idx, (int, slice, list, np.ndarray)):
            raise ValueError(
                "`e_idx` must be of type `int`, `slice` or an index array."
            )

        _v2e = v2e[:, e_idx]

    if return_sparse is True:
        return _v2e

    v_idx = np.vstack(np.split(_v2e.indices, _v2e.indptr)[1:-1])
    v_sgn = np.vstack(np.split(_v2e.data, _v2e.indptr)[1:-1])

    return v_idx, v_sgn


def vtx_to_edge(
    v2e: csr_matrix, v_idx: ArrayLike | slice | None = None, return_sparse: bool = False
) -> List[ArrayLike]:
    """
    Get edge indices from given vertex indices.

    Parameters
    ----------
    v2e : scipy.sparse.csr_matrix
        A vertex-to-edge incidence matrix in `csr` format.
    v_idx : int, slice, np.ndarray
        Vertex index or indices to get their edge indices.
    return_sparse : bool
        Whether to return a sparse matrix.

    Returns
    -------
    e_idx : List[np.ndarray]
        A list with arrays with the edge indices for each
        vertex.
    """
    if not isinstance(v2e, csr_matrix):
        v2e = v2e.tocsr()

    if v_idx is None:
        _v2e = v2e

    else:
        if not isinstance(v_idx, (int, slice, list, np.ndarray)):
            raise ValueError(
                "`v_idx` must be of type `int`, `slice` or an index array."
            )

        _v2e = v2e[v_idx, :]

    if return_sparse is True:
        return _v2e

    return np.split(_v2e.indices, _v2e.indptr)[1:-1]


def face_to_vtx(
    v2f: csc_matrix, f_idx: ArrayLike | slice | None = None, return_sparse: bool = False
) -> ArrayLike:
    """
    Get vertex indices from face index.

    Parameters
    ----------
    v2f : scipy.sparse.csc_matrix
        A vertex-to-face incidence matrix in `csc` format.
    f_idx : int, slice, np.ndarray
        Face index or indices to get their vertices.
    return_sparse : bool
        Whether to return a sparse matrix.

    Returns
    -------
    np.ndarray
        An array of shape `(n_f, 3)` with the vertex indices
        of each face, where `n_f` the number of faces.
    """
    if not isinstance(v2f, csc_matrix):
        v2f = v2f.tocsc()

    if f_idx is None:
        _v2f = v2f

    else:
        if not isinstance(f_idx, (int, slice, list, np.ndarray)):
            raise ValueError(
                "`f_idx` must be of type `int`, `slice` or an index array."
            )

        _v2f = v2f[:, f_idx]

    if return_sparse is True:
        return _v2f

    v_idx = np.vstack(np.split(_v2f.indices, _v2f.indptr)[1:-1])

    return v_idx


def vtx_to_face(
    v2f: csr_matrix, v_idx: ArrayLike | slice | None = None, return_sparse: bool = False
) -> List[ArrayLike]:
    """
    Get face indices from vertex index.

    Parameters
    ----------
    v2f : scipy.sparse.csr_matrix
        A vertex-to-face incidence matrix in `csr` format.
    v_idx : int, slice, np.ndarray
        Vertex index or indices to get their associating faces.
    return_sparse : bool
        Whether to return a sparse matrix.

    Returns
    -------
    List[np.ndarray]
        A list with arrays with face indices for each given
        vertex index.
    """
    if not isinstance(v2f, csr_matrix):
        v2f = v2f.tocsr()

    if v_idx is None:
        _v2f = v2f

    else:
        if not isinstance(v_idx, (int, slice, list, np.ndarray)):
            raise ValueError(
                "`v_idx` must be of type `int`, `slice` or an index array."
            )
        _v2f = v2f[v_idx, :]

    if return_sparse is True:
        return _v2f

    return np.split(_v2f.indices, _v2f.indptr)[1:-1]
