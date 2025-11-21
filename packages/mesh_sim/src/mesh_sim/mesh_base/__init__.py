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

from .ops import (
    calculate_face_normals,
    calculate_face_area,
    calculate_face_centroids,
    calculate_face_edges_lengths,
    calculate_edges_opposite_angles,
    calculate_vertex_normals,
)

from .utils import (
    build_signed_incidence_matrices,
    build_vertex_to_face_incidence_matrix,
)

from .operators import (
    operator_laplace_beltrami,
    operator_dirac_relative,
)
