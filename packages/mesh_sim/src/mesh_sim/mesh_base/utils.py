from typing import Tuple, List

import scipy
import itertools

from scipy.sparse import csc_matrix


def build_signed_incidence_matrices(
    vertices: List[float], faces: List[List[int]]
) -> Tuple[csc_matrix, csc_matrix, int, int, int]:
    """
    Build indidence matrices from list of vertices / faces.

    Parameters
    ----------
    vertices : List[float]
        A list of vertex coordinates.
    faces : List[List[int]
        A list of vertex indices in counter-clockwise order.

    Returns
    -------
    v2e : scipy.sparse.csc_matrix
        A sparse incidence matrix from vertices to edges in
        `csc` format.
    e2f : scipy.sparse.csc_matrix
        A sparse incidence matrix from edges to faces in `coo`
        format.
    n_v : int
        The number of vertices.
    n_e : int
        Total number of directed(?) edges (or half-edges ?).
    n_f : int
        Total number of faces.
    """
    # Counter for edge indices
    e_cntr = itertools.count()

    n_v, n_f = len(vertices), len(faces)

    # v2e: (data, v_s, e_s)
    v2e__d, v2e__v, v2e__e = [], [], []
    # e2f: (data, e_s, f_s)
    e2f__d, e2f__e, e2f__f = [], [], []

    # `e_s` is a helper variable for bookeeping
    e_s = {}

    for f_idx, face in enumerate(faces):

        for i in range(3):

            v_from, v_to = face[i], face[(i + 1) % 3]

            e_key = tuple(sorted((v_from, v_to)))

            # If edge does not exist in edges:
            # - Create new edge index
            # - Populate `vertex_to_edge` matrix.
            if e_key not in e_s:
                e_idx = next(e_cntr)
                e_s[e_key] = (e_idx, (v_from, v_to))

                v2e__d.append(-1)
                v2e__v.append(v_from)
                v2e__e.append(e_idx)

                v2e__d.append(1)
                v2e__v.append(v_to)
                v2e__e.append(e_idx)

                # When we add a new edge, it is always consistent
                consistent = True

            else:
                e_idx, edge = e_s[e_key]

                # Check if the existing edge is flipped
                consistent = edge == (v_from, v_to)

            # Populate `edge_to_face`.
            e2f__d.append(1 if consistent else -1)
            e2f__e.append(e_idx)
            e2f__f.append(f_idx)

    # Add 1 to get the number of edges as the
    # edge index starts from zero.
    n_e = len(e_s)

    v2e = scipy.sparse.coo_matrix((v2e__d, (v2e__v, v2e__e)), (n_v, n_e), dtype="int8")
    e2f = scipy.sparse.coo_matrix((e2f__d, (e2f__e, e2f__f)), (n_e, n_f), dtype="int8")

    return v2e.tocsc(), e2f.tocsc(), n_v, n_e, n_f


def build_vertex_to_face_incidence_matrix(
    v2e: csc_matrix, e2f: csc_matrix
) -> csc_matrix:
    """
    Create a vertex-to-face incidence matrix.

    Parameters
    ----------
    v2e : scipy.sparse.csc_matrix
        A vertex-to-edge incidence matrix in `csc` format.
    e2f : scipy.sparse.csc_matrix
        An edge-to-face incidence matrix in `csc` format.

    Returns
    -------
    v2f : scipy.sparse.csc_matrix
        A vertex-to-face incidence matrix in `csc` format.
    """
    if not isinstance(v2e, csc_matrix):
        v2e = v2e.tocsc()
    if not isinstance(e2f, csc_matrix):
        e2f = e2f.tocsc()

    v2f = abs(v2e).dot(abs(e2f))

    v2f.data.fill(1)
    
    v2f.sort_indices()

    return v2f


# NOT USED
def build_double_edge_incidence_matrices(
    vertices: List[float], faces: List[List[int]]
) -> Tuple[csc_matrix, csc_matrix, int, int, int]:
    """
    Build indidence matrices from list of vertices / faces.

    Parameters
    ----------
    vertices : List[float]
        A list of vertex coordinates.
    faces : List[List[int]
        A list of vertex indices in counter-clockwise order.

    Returns
    -------
    v2e : scipy.sparse.csc_matrix
        A sparse incidence matrix from vertices to edges in
        `csc` format.
    e2f : scipy.sparse.csc_matrix
        A sparse incidence matrix from edges to faces in `coo`
        format.
    n_v : int
        The number of vertices.
    n_e : int
        Total number of directed(?) edges (or half-edges ?).
    n_f : int
        Total number of faces.
    """
    # Directed edges
    e_cntr = itertools.count()

    n_v, n_f = len(vertices), len(faces)

    v2e__d, v2e__v, v2e__e = [], [], []
    e2f__d, e2f__e, e2f__f = [], [], []

    for f_idx, face in enumerate(faces):

        for i in range(3):

            v_from, v_to = face[i], face[(i + 1) % 3]

            e_idx = next(e_cntr)

            # Populate vertex_to_edge
            # v2e.append((1, (v_from, e_idx)))
            # v2e.append((1, (v_to, e_idx)))
            v2e__d.append(1)
            v2e__d.append(1)
            v2e__v.append(v_from)
            v2e__v.append(v_to)
            v2e__e.append(e_idx)
            v2e__e.append(e_idx)

            # Populate edge_to_face
            # e2f.append((1, (e_idx, f_idx)))
            e2f__d.append(1)
            e2f__e.append(e_idx)
            e2f__f.append(f_idx)

    # Add 1 to get the number of edges as the
    # edge index starts from zero.
    n_e = e_idx + 1

    v2e = scipy.sparse.coo_matrix((v2e__d, (v2e__v, v2e__e)), (n_v, n_e), dtype="int8")
    e2f = scipy.sparse.coo_matrix((e2f__d, (e2f__e, e2f__f)), (n_e, n_f), dtype="int8")

    return v2e.tocsc(), e2f.tocsc(), n_v, n_e, n_f
