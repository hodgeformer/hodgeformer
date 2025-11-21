import pytest

from mesh_sim.mesh_base import (
    build_signed_incidence_matrices,
    build_vertex_to_face_incidence_matrix,
)

# Simple example
VERTICES = [
    [-1.0, 0.0, 0.0],
    [0.0, -0.66, 0.0],
    [0.0, 0.66, 0.0],
    [1.0, 0.0, 0.0],
]
FACES = [
    [1, 0, 2],
    [2, 3, 1],
]

EXAMPLES = [[VERTICES, FACES]]


def test_build_signed_incidence_matrices():
    """ """
    v2e, e2f, n_v, n_e, n_f = build_signed_incidence_matrices(VERTICES, FACES)

    assert n_v == len(VERTICES)
    assert n_f == len(FACES)

    # Chaining two boundary operators results to 0.
    assert v2e.dot(e2f).nnz == 0


def test_build_vertex_to_face_incidence_matrix():
    """ """
    v2e, e2f, _, _, _ = build_signed_incidence_matrices(VERTICES, FACES)

    v2f = build_vertex_to_face_incidence_matrix(v2e, e2f)

    assert v2f.shape[0] == v2e.shape[0]
    assert v2f.shape[1] == e2f.shape[1]
