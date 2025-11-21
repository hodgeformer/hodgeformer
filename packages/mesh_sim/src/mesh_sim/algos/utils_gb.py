try:
    import graphblas as gb
except ImportError:
    print("`graphblas` package is not available, cannot run method.")

from graphblas import Matrix, Vector, Scalar
from scipy.sparse import csc_matrix, csr_matrix


def inc_to_deg_matrix(inc_matrix: Matrix) -> Matrix:
    """
    Get a degree matrix from an incidence matrix. Reduction takes place row-wise.

    Parameteres
    -----------
    inc_matrix : gb.Matrix
        Incidence matrix of shape `(n, m)` taking `m` to `n` elements,
        e.g. `m` edges to `n` vertices.

    Returns
    -------
    gb.Matrix
        Degree matrix of shape `(n, n)`.
    """
    deg_vec = gb.op.abs(inc_matrix).reduce_rowwise()

    return gb.Matrix.diag(deg_vec)


def inc_to_lap_matrix(inc_matrix: Matrix) -> Matrix:
    """
    Get a laplacian matrix from an incidence matrix.

    Parameteres
    -----------
    inc_matrix : gb.Matrix
        Incidence matrix of shape `(n, m)` taking `m` to `n` elements,
        e.g. `m` edges to `n` vertices.

    Returns
    -------
    gb.Matrix
        Laplacian matrix of shape `(n, n)`.
    """
    return gb.semiring.plus_times(inc_matrix @ inc_matrix.T)


def inc_to_line_graph_adj_matrix(inc_matrix: Matrix) -> Matrix:
    """
    Build the adjacency matrix from a boundary operator.
    """
    # Create the unoriented adjacency matrix
    inc_matrix = gb.op.abs(inc_matrix)

    deg_matrix = inc_to_deg_matrix(inc_matrix)

    lap_matrix = inc_to_lap_matrix(inc_matrix)

    return (lap_matrix - deg_matrix).select(gb.select.valuene, 0).new()


def inc_to_adj_matrix(inc_matrix: Matrix) -> Matrix:
    """
    Build the adjacency matrix from a coboundary operator.
    """
    deg_matrix = inc_to_deg_matrix(inc_matrix)

    lap_matrix = inc_to_lap_matrix(inc_matrix)

    return (deg_matrix - lap_matrix).select(gb.select.valuene, 0).new()


def scipy_to_gb(scp_matrix: csr_matrix | csc_matrix) -> Matrix:
    """
    Convert `scipy` matrix to `graphblas`.

    Parameters
    ----------
    scp_matrix : csr_matrix | csc_matrix
        A `scipy` sparse matrix of 'csr' or 'csc' format.

    Returns
    -------
    gb_matrix : gb.Matrix
        Matrix as `graphblas` instance.
    """
    if isinstance(scp_matrix, csr_matrix):
        gb_matrix = gb.io.from_scipy_sparse(scp_matrix)
    elif isinstance(scp_matrix, csc_matrix):
        gb_matrix = gb.io.from_scipy_sparse(scp_matrix)
    else:
        raise ValueError("Input matrix must be a sparse scipy matrix.")

    return gb_matrix
