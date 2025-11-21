from typing import Tuple
from numpy.typing import ArrayLike

import numpy as np

import scipy

from scipy.sparse import (
    csc_matrix,
    csr_matrix,
    bsr_matrix,
    coo_matrix,
)

from .maps import (
    vtx_to_face,
    face_to_edge,
    face_to_vtx,
)

from .ops import (
    calculate_face_area,
    calculate_face_edges_lengths,
    calculate_edges_opposite_angles,
    calculate_face_normals,
    calculate_vertex_normals,
)


def _cotangent(thetas: ArrayLike) -> ArrayLike:
    """
    Get cotangent(theta).
    """
    return np.where(thetas != 0, -np.tan(thetas + np.pi / 2), 0)


def calculate_edge_cotangent_weights(
    v_s: ArrayLike, v2e: csc_matrix, e2f: csc_matrix
) -> ArrayLike:
    """
    For each edge calculate its cotangent weight.
    """
    # Array of `(n_f, 3)` with the opposite angle for each edge.
    e_angles_per_face = calculate_edges_opposite_angles(v_s, v2e, e2f, f_idx=None)

    # Angle to cotangent weights.
    e_cotans = _cotangent(e_angles_per_face)

    e_cotans = e_cotans.sum(axis=-1)

    return 0.5 * e_cotans


def operator_laplace_beltrami(
    v_s: ArrayLike,
    v2e: csc_matrix,
    e2f: csc_matrix,
    eps: float = 1e-8,
) -> csr_matrix:
    """
    Build a Laplace Beltrami operator.
    """
    e_weights = calculate_edge_cotangent_weights(v_s, v2e, e2f)
    # Can we calculate the cotangent Laplacian with a simple dot product?
    #
    # Make sparse diagonal matrix of shape `(n_e, n_e)` where each value
    # on the diagonal corresponds to the cotangent weight for this edge.

    e_weights = scipy.sparse.diags(e_weights, format="csc")

    # Calculate the cotangent Laplacian with a single sparse dot product

    # Need to verify with construct per vtx(?)
    # v_idx = vtx_to_edge(v2e=v2e, v_idx=None)

    return v2e.dot(e_weights).dot(v2e.T)


def get_transformation_matrix_quaternion_to_real():
    """
    Form real 4x4 matrix from quaternion.
    ```
    a, b, c, d = x.tolist()

    return np.array(
        [
            [a, -b, -c, -d],
            [b, a, -d, c],
            [c, d, a, -b],
            [d, -c, b, a],
        ]
    )
    ```
    """

    return np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype="float32",
    )


def operator_dirac_relative(
    v_s: ArrayLike, v2e: csc_matrix, e2f: csc_matrix, v2f: csc_matrix
) -> csc_matrix:
    """
    Construct the discrete quaternionic relative dirac operator for a
    3d triangle mesh (as described by "A Dirac Operator for Extrinsic
    Shape Analysis" [Liu, Jacobson, and Crane. 2017]).

    From Liu & Crane:
        See `https://www.dgp.toronto.edu/projects/dirac/`
        See `https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Crane11.pdf`
        See `https://github.com/alecjacobson/gptoolbox/blob/master/mesh/dirac_operator.m`

    And from `Surface Networks`:
        See `https://arxiv.org/abs/1705.10819`
        See `https://github.com/jiangzhongshi/SurfaceNetworks/blob/dev/src/utils/mesh.py#L35`

    Parameters
    ----------
    v_s : np.ndarray
        The mesh vertices coordinates.
    v2f : scipy.sparse.csc_matrix
        A vertex-to-face incidence matrix in `csc` format.

    Returns
    -------
    csc_matrix
        A sparse matrix of shape `(4*n_f x 4*n_v)`.
    """
    n_v, n_f = v2f.shape

    # `(n_f,)`
    f_area = calculate_face_area(v_s, v2e, e2f, f_idx=None)

    # `(n_v, 3)`
    v_normals = calculate_vertex_normals(v_s=v_s, v2e=v2e, e2f=e2f, v2f=v2f)

    # `(n_f, 3)`
    v_idx = face_to_vtx(v2f)

    # Cycle shifted normals per face (face, v_idx, normal_coordinate)
    N_j = v_normals[v_idx[:, [1, 2, 0]]]
    N_k = v_normals[v_idx[:, [2, 0, 1]]]

    # `(n_f, 3, 3)`
    dN = N_k - N_j

    # `(n_f, 3, 4)` (Add zero coordinate)
    dN = np.pad(dN, ((0, 0), (0, 0), (1, 0)))
    dN = -dN / (2 * f_area.reshape(n_f, 1, 1))

    Q = get_transformation_matrix_quaternion_to_real()

    # `(n_f, 3, 4, 4)`
    dN_q = np.einsum("ijk, lk -> ijl", dN, Q).reshape(n_f, 3, 4, 4)

    # `(4 * n_v, 4 * n_f)`
    dirac = _build_dirac_sparse_bsr(v2f, dN_q)

    return dirac.tocsc()


def _build_dirac_sparse_bsr(v2f, dN_q):
    """
    Build a sparse matrix for the relative dirac operator.
    """
    n_v, n_f = v2f.shape

    # When converting `csc` to `csr` the data along with the
    # indices are reordered. To keep track data position we
    # can change the `v2f.data` with an index array. After
    # conversion to `csr` the `v2f__csr.data` can be used as
    # an indexing array for the `dN_q` block matrices.
    #
    # This is happening because artifacts such as `dN_q` have
    # been calculated with `v2f` csc matrix and follow its
    # order.
    #
    # To save memory we add the index to the `v2f` matrix and
    # not to a copy, so we have to restore the original data
    # (that is all ones).

    v2f.data = np.arange(0, len(v2f.data), dtype="int32")

    v2f__csr = v2f.tocsr()

    v2f.data.fill(1)

    data_idx, indices, indptr = v2f__csr.data, v2f__csr.indices, v2f__csr.indptr

    return bsr_matrix(
        (dN_q.reshape(n_f * 3, 4, 4)[data_idx], indices, indptr),
        shape=(4 * n_v, 4 * n_f),
    )


def _build_dirac_sparse_coo_vectorized(v2f, dN_q):
    """
    Build a sparse matrix for the relative dirac operator.
    """
    n_v, n_f = v2f.shape

    v2f__coo = v2f.tocoo()

    # Expand rows & cols indices, vectorized solution
    rows = np.repeat(4 * v2f__coo.row, 16)

    row_indices = np.arange(rows.size).reshape(rows.size // 4, 4)

    for i in range(1, 4):
        np.add.at(rows, row_indices[np.arange(i, rows.size // 4, 4)], i)

    cols = np.repeat(4 * v2f__coo.col, 16)

    for i in range(1, 4):
        np.add.at(cols, np.arange(i, cols.size, 4), i)

    return coo_matrix((dN_q.flatten(), (rows, cols)), shape=(4 * n_v, 4 * n_f))


def _build_dirac_sparse_coo_unvectorized(v2f, dN_q):
    """
    Build a sparse matrix for the relative dirac operator.
    """
    n_v, n_f = v2f.shape

    v2f__coo = v2f.tocoo()

    # Expand rows & cols indices, unvectorized solution
    rows = v2f__coo.row
    cols = v2f__coo.col
    new_rows, new_cols = [], []

    for row, col in zip(rows, cols):

        row = 4 * row
        col = 4 * col

        for row_idx in range(4):

            for col_idx in range(4):

                new_rows.append(row + row_idx)
                new_cols.append(col + col_idx)

    new_rows = np.array(new_rows, dtype="int32")
    new_cols = np.array(new_cols, dtype="int32")

    return coo_matrix((dN_q.flatten(), (new_rows, new_cols)), shape=(4 * n_v, 4 * n_f))


def mass_matrix__vertex_lumped(
    v_s: ArrayLike, v2e: csc_matrix, e2f: csc_matrix, v2f: csc_matrix
) -> ArrayLike:
    """
    Calculate a mass matrix `M` using the vertex lumping method.

    The mass matrix `M`, is a sparse symmetric matrix depending
    only on the area of the faces of the mesh. The vertex lumped
    mass matrix is calculated by calculating for each vertex the
    area of its incident faces (or a.k.a. 'barycentric cell area').

            M_vertex_lumped_ii = 1/3 * Sum_over_ijk Aijk

    Parameters
    ----------
    v_s : np.ndarray
        The mesh vertices coordinates.
    v2f : scipy.sparse.csc_matrix
        A vertex-to-face incidence matrix in `csc` format.

    Returns
    -------
    csc_matrix
        A sparse diagonal matrix in `csc` format with the lumped
        mass per vertex.
    """
    f_area = calculate_face_area(v_s, v2e, e2f, f_idx=None)

    f_indices = vtx_to_face(v2f)

    mass = np.array([(1.0 / 3) * np.mean(f_area[f_idx]) for f_idx in f_indices])

    return scipy.sparse.diags(mass, format="csc")


def mass_matrix__galerkin():
    """ """
    pass


def compute_hks(evals: ArrayLike, evecs: ArrayLike, scales: ArrayLike) -> ArrayLike:
    """
    Estimate `hks` features.

    Parameters
    ----------
    evals : ArrayLike
        The eigenvalues of shape (K,).
    evecs : ArrayLike
        The eigenvector values of shape (V, K) values.
    scales : ArrayLike
        The time scales of shape (S,).

    Returns
    -------
    ArrayLike
        The calculate hks values of shape (V, S).
    """
    coefs = np.exp(-evals[np.newaxis, :] * scales[:, np.newaxis])  # (1,S,K)

    terms = coefs[np.newaxis, ...] * (evecs**2)[:, np.newaxis, :]  # (V,S,K)

    return np.sum(terms, axis=-1)  # (V,S)


def compute_eigen(
    v_s: ArrayLike, v2e: csc_matrix, e2f: csc_matrix, v2f: csc_matrix, k: int = 128
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Compute Laplacian eigenvectors & eigenvalues.
    """
    eps = 1e-8

    # import potpourri3d as pp3d

    # L = pp3d.cotan_laplacian(verts_np, faces_np, denom_eps=1e-10)
    # massvec_np = pp3d.vertex_areas(verts_np, faces_np)
    # massvec_np += eps * np.mean(massvec_np)

    n_v = len(v_s)

    L = operator_laplace_beltrami(v_s, v2e, e2f)
    L = L + scipy.sparse.identity(n_v) * (eps)

    M = mass_matrix__vertex_lumped(v_s, v2e, e2f, v2f)
    eps = 1e-8

    # Got from Sharp et al. (DiffusionNet)
    fail_count = 0

    while True:
        try:
            # We would be happy here to lower tol or maxiter since we don't need these
            # to be super precise, but for some reason those parameters seem to have no effect
            evals, evecs = scipy.sparse.linalg.eigsh(L, k=k, M=M, sigma=eps)

            # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
            evals = np.clip(evals, a_min=0.0, a_max=float("inf"))

            break
        except Exception as e:
            print(e)
            if fail_count > 3:
                raise ValueError("failed to compute eigendecomp")
            fail_count += 1
            print("--- decomp failed; adding eps ===> count: " + str(fail_count))
            L = L + scipy.sparse.identity(L.shape[0]) * (eps * 10**fail_count)

    return evals, evecs


def compute_hks_autoscale(
    v_s: ArrayLike, v2e: csc_matrix, e2f: csc_matrix, v2f: csc_matrix, num: int
) -> ArrayLike:
    """
    Compute `hks` features.
    """
    evals, evecs = compute_eigen(v_s, v2e=v2e, e2f=e2f, v2f=v2f, k=128)

    # these scales roughly approximate those suggested in the hks paper
    scales = np.logspace(-2.0, 0.0, num=num, dtype=evals.dtype)

    return compute_hks(evals, evecs, scales)
