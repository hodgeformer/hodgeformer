from typing import List, Tuple

import hashlib
import numpy as np

from .mesh_base.utils import (
    build_signed_incidence_matrices,
    build_vertex_to_face_incidence_matrix,
)

from .mesh_base.ops import (
    get_edge_coords_ordered,
    get_edge_coords_unordered,
    get_face_coords_ordered,
    get_face_coords_unordered,
)

from .mesh_base.ops import (
    calculate_face_centroids,
    calculate_face_area,
    calculate_face_normals,
    calculate_face_edges_lengths,
    calculate_edges_opposite_angles,
    calculate_edge_lengths,
    calculate_edge_normals,
    calculate_vertex_cell_area,
    calculate_vertex_normals,
)

from .mesh_base.operators import (
    operator_laplace_beltrami,
    operator_dirac_relative,
)

from .utils.open3d import build_o3d_mesh


class MeshSignedIncidenceMatrices:
    """
    The `Mesh` class represents a mesh instance.
    """

    def __init__(self, vertices, faces, uid=None, mesh_o3d=None):
        """
        Initializer for a `Mesh` instance.
        """
        self.uid = uid

        self.v_s = _validate_vertices(vertices)
        self.f_s = _validate_faces(faces)

        self.v2e__csc, self.e2f__csc, self.n_v, self.n_e, self.n_f = (
            build_signed_incidence_matrices(vertices, faces)
        )

        self.v2f__csc = build_vertex_to_face_incidence_matrix(
            self.v2e__csc, self.e2f__csc
        )

        if mesh_o3d is None:
            self.mesh_o3d = build_o3d_mesh(vertices=vertices, faces=faces)
        else:
            self.mesh_o3d = mesh_o3d

        # self.mesh_o3d.orient_triangles()
        self.mesh_o3d.compute_triangle_normals()
        self.mesh_o3d.compute_vertex_normals()

        self._cache_csr_matrices()

    def __hash__(self):
        """
        Hash method for mesh class.
        """
        if self.uid is None:
            raise ValueError("Cannot implement a thred safe hash")

        uid_hash = hashlib.md5(self.uid.encode())

        return int(uid_hash.hexdigest(), 16)

    def __eq__(self, other):
        if isinstance(other, MeshSignedIncidenceMatrices):
            return self.uid == other.uid
        else:
            return self.uid == other

    def face_centroids(self, f_idx=None):
        """
        Calculate face centroids.
        """
        return calculate_face_centroids(v_s=self.v_s, v2f=self.v2f__csc, f_idx=f_idx)

    def face_area(self, f_idx=None):
        """
        Calculate face area.
        """
        return calculate_face_area(
            v_s=self.v_s, v2e=self.v2e__csc, e2f=self.e2f__csc, f_idx=f_idx
        )

    def edge_normals_from_faces(self, e_idx=None):
        """
        Calculate edge normals.
        """
        return calculate_edge_normals(
            v_s=self.v_s,
            v2e=self.v2e__csc,
            e2f=self.e2f__csc,
            e_idx=e_idx,
            base_normals=self.face_normals(),
            base_elem="f",
        )

    def edge_normals_from_vertices(self, e_idx=None):
        """
        Calculate edge normals.
        """
        return calculate_edge_normals(
            v_s=self.v_s,
            v2e=self.v2e__csc,
            e2f=self.e2f__csc,
            e_idx=e_idx,
            base_normals=self.vertex_normals(),
            base_elem="v",
        )

    def face_normals(self, f_idx=None):
        """
        Calculate face normals.
        """
        if hasattr(self, "mesh_o3d"):
            return np.asarray(self.mesh_o3d.triangle_normals)

        else:
            return calculate_face_normals(
                v_s=self.v_s, v2e=self.v2e__csc, e2f=self.e2f__csc, f_idx=f_idx
            )

    def face_edges_lengths(self, f_idx=None):
        """
        Calculate face edge lengths.
        """
        return calculate_face_edges_lengths(
            v_s=self.v_s, v2e=self.v2e__csc, e2f=self.e2f__csc, f_idx=f_idx
        )

    def face_edges_opposite_angles(self, f_idx=None):
        """
        Calculate face edge opposite angles.
        """
        return calculate_edges_opposite_angles(
            v_s=self.v_s, v2e=self.v2e__csc, e2f=self.e2f__csc, f_idx=f_idx
        )

    def face_coords(self, f_idx=None, ordered=True):
        """
        Get face coords.
        """
        if ordered is False:
            return get_face_coords_unordered(
                v_s=self.v_s, v2f=self.v2f__csc, f_idx=f_idx
            )
        else:
            return get_face_coords_ordered(
                v_s=self.v_s, v2e=self.v2e__csc, e2f=self.e2f__csc, f_idx=f_idx
            )

    def edge_lengths(self, e_idx=None):
        """
        Calculate edge lengths.
        """
        return calculate_edge_lengths(v_s=self.v_s, v2e=self.v2e__csc, e_idx=e_idx)

    def edge_coords(self, e_idx=None, ordered=True):
        """
        Get face coords.
        """
        if ordered is False:
            return get_edge_coords_unordered(
                v_s=self.v_s, v2e=self.v2e__csc, e_idx=e_idx
            )
        else:
            return get_edge_coords_ordered(v_s=self.v_s, v2e=self.v2e__csc, e_idx=e_idx)

    def vertex_cell_area(self, v_idx=None):
        """
        Calculate vertex cell area.
        """
        return calculate_vertex_cell_area(
            v_s=self.v_s,
            v2e=self.v2e__csc,
            e2f=self.e2f__csc,
            v2f=self.v2f__csc,
            v_idx=v_idx,
        )

    def vertex_normals(self, v_idx=None, how="area_weighted"):
        """
        Calculate vertex normals.
        """
        if hasattr(self, "mesh_o3d"):
            return np.asarray(self.mesh_o3d.vertex_normals)

        return calculate_vertex_normals(
            v_s=self.v_s,
            v2e=self.v2e__csc,
            e2f=self.e2f__csc,
            v2f=self.v2f__csc,
            v_idx=v_idx,
            how=how,
            f_normals=self.face_normals(),
        )

    def _cache_geometrical_attrs(self):
        """
        Calculate and cache geometrical attributes of interest.
        """
        self._cache = {
            "face__area": None,
            "face__normals": None,
            "face__centroids": None,
            "face__edge_lengths": None,
            "face__edge_opposite_angles": None,
            "vertex__normals": None,
        }

        self._cache["face__area"] = self.face_area()
        self._cache["face__normals"] = self.face_normals()
        self._cache["face__centroids"] = self.face_centroids()
        self._cache["face__edge_lengths"] = self.face_edges_lengths()
        self._cache["face__edge_opposite_angles"] = self.face_edges_opposite_angles()
        self._cache["edge__lengths"] = self.edge_lengths()
        self._cache["vertex__normals"] = self.vertex_normals()

    def _cache_csr_matrices(self):
        """
        Cache boundary operators in `csr_format` to facilitate calculations.
        """
        self.v2e__csr = self.v2e__csc.tocsr()
        self.e2f__csr = self.e2f__csc.tocsr()
        self.v2f__csr = self.v2f__csc.tocsr()

    def __str__(self):
        """
        String representation for `MeshSignedIncidenceMatrices` instance.
        """
        return "Mesh with vertices: {}, edges: {}, faces: {}".format(
            self.n_v, self.n_e, self.n_f
        )

    @classmethod
    def from_mesh_o3d(cls, mesh_o3d, uid):
        """
        Create instance from a `open3d.geometry.TriangleMesh` instance.
        """
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)

        instance = cls(vertices, faces, uid=uid, mesh_o3d=mesh_o3d)

        return instance


def _validate_vertices(vertices):
    """
    Validate input vertices dtype.
    """
    if not isinstance(vertices, np.ndarray):
        return np.array(vertices, dtype="float32")

    elif getattr(vertices, "dtype") != "float32":
        return np.array(vertices, dtype="float32")

    else:
        return vertices


def _validate_faces(faces):
    """
    Validate input faces dtype.
    """
    if not isinstance(faces, np.ndarray):
        return np.array(faces, dtype="int32")

    elif getattr(faces, "dtype") != "int32":
        return np.array(faces, dtype="int32")

    else:
        return faces
