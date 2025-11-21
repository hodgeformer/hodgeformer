import pytest

from mesh_sim.mesh import MeshSignedIncidenceMatrices

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


class TestMeshSignedIncidenceMatrices:

    @pytest.fixture(scope="function")
    def mesh(self):
        return MeshSignedIncidenceMatrices(VERTICES, FACES)

    def test_mesh_inner_elements(self, mesh):
        assert mesh.n_v == 4
        assert mesh.n_e == 5
        assert mesh.n_f == 2

    def test_mesh_inner_matrices(self, mesh):
        assert mesh.e2f__csc.shape == (5, 2)
        assert mesh.v2e__csc.shape == (4, 5)
