from numpy.typing import ArrayLike

import itertools

import numpy as np

from scipy.sparse import csc_matrix, dia_matrix, diags_array

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


def get_face_edge_neighbors():
    """ """
    pass


def get_face_vertex_neighbor():
    """ """
    pass


def get_vertex_edge_neighbors():
    """ """
    pass


def get_vertex_face_neighbors():
    """ """
    pass


def build_deg_matrix_from_incidence(v2e: csc_matrix):
    """
    Build the degree matrix from a boundary operator.
    """
    return abs(v2e).sum(axis=1)


def build_adj_matrix_from_incidence(v2e: csc_matrix, normalize=True, renorm=False):
    """
    Build the adjacency matrix from a boundary operator.
    """
    deg = build_deg_matrix_from_incidence(v2e)

    lap = v2e @ v2e.T

    adj = dia_matrix((deg.squeeze(), 0), shape=lap.shape) - lap

    if renorm is True:
        identity = diags_array([1.0], shape=adj.shape, dtype="float32", format="csc")
        adj = adj + identity
        deg = np.add(deg + 1, dtype="float32")

    if normalize is True:
        deg_inv = np.reciprocal(np.sqrt(deg_inv))
        deg_inv[deg == 0] = 0.0

        diag = dia_matrix((deg_inv.squeeze(), 0), shape=adj.shape)
        adj = diag @ adj @ diag

    return adj
