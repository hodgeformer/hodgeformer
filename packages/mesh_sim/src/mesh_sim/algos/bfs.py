"""
Documentation for non-documented online `graphblas` functionality (taken from github):

* `ss.selectk`

how : {'first', 'last', 'random'}
    Where 'random' means to choose k elements with equal probability, 'first'
    to choose the first k elements and 'last' to choose the last k elements.
k : int
    The number of elements to choose from each row


* `ss.compactify`

Push all values to the left of a sparse matrix.

how : {'first', 'last', 'random', 'smallest', 'largest'}
    How to compress the values:
    - first : take the values furthest to the left (or top)
    - last : take the values furthest to the right (or bottom)
    - smallest : take the smallest values (if tied, may take any)
    - largest : take the largest values (if tied, may take any)
    - random : take values randomly with equal probability and without replacement
        Chosen values may not be ordered randomly
k : int
    The number of columns (or rows) of the returned Matrix.  If not specified,
    then the Matrix will be "compacted" to the smallest ncols (or nrows) that
    doesn't lose values.
order : {'rowwise', 'columnwise'}
    Whether to compactify rowwise or columnwise. Rowwise shifts all values
    to the left, and columnwise shifts all values to the top.
reverse : bool
    Reverse the values in each row (or column) when True
as_index : bool
    Return the column (or row) index of the value when True. If there are ties
    for "smallest" and "largest", then any valid index may be returned.
"""

from typing import Tuple, List
from numpy.typing import ArrayLike

import numpy as np

from scipy.sparse import csc_matrix, csr_matrix

try:
    import graphblas as gb

    from graphblas import Matrix
except ImportError:
    print("`graphblas` package is not available, cannot run method.")

from .utils_gb import (
    scipy_to_gb,
    inc_to_adj_matrix,
    inc_to_line_graph_adj_matrix,
)


def bigbird_neighbors(
    inc_matrix: csr_matrix | csc_matrix,
    inc_type: str = "v2e",
    n: int | None = None,
    k: int = 3,
    dilations: int = 1,
    modes: List[str] = ["g", "r", "l"],
    percs: List[float | int] = [4, 0.2, 0.7],
    rng: np.random.RandomState | None = None,
):
    """
    Collect connections following the strategy proposed by the BigBird transformer.

    The BigBird transformer defines the following types of attention connections:
    - local connections
    - random connections
    - global connections

    Parameters
    ----------
    inc_matrix : csr_matrix | csc_matrix
        An incidence matrix representing a graph.
    inc_type : {'v2e', 'e2v', 'e2f'}
        The type of input incidence matrix.
    n : int
        Number of connections to collect.
    k : int
        Number of hops / bfs levels to reach.
    dilations : int
        Dilations of bfs hops.
    modes : List[str]
        The type of connections 'g': global, 'r': random and 'l': local
        to use when extracting a big-bird neighborhood.
    percs : List[float | int]
        Percentage or absolute number of items to correspond to selected modes when
        extracting a big-bird neighborhood.

    Returns
    -------
    np.ndarray
        An array of shape `(l, n)` where `l` the number of rows of the incidence matrix
        and `n` the number of connections.
    """
    if rng is None:
        rng = np.random.RandomState()

    nrows, _ = inc_matrix.shape

    n_map = dict(zip(modes, percs))

    matrices = []

    if "g" in modes:
        n_global = int(n_map["g"] * n) if isinstance(n_map["g"], float) else n_map["g"]

        glbl_matrix = get_global_matrix(nrows, n_global, rng)

        matrices.append(glbl_matrix)

    if "r" in modes:
        n_random = (
            int(round(n_map["r"] * n)) if isinstance(n_map["r"], float) else n_map["r"]
        )

        rand_matrix = get_random_matrix(nrows, n_random, rng, sym=True)

        matrices.append(rand_matrix)

    if "l" in modes:
        _n = int(round(n_map["l"] * n)) if isinstance(n_map["l"], float) else n_map["l"]

        bfs_matrix = get_bfs_matrix(
            inc_matrix=inc_matrix, inc_type=inc_type, k=k, dilations=dilations, _max=_n
        )

        matrices.append(bfs_matrix)

    # Aggregate glbl_matrix, rand_matrix, dist_matrix
    bbrd_matrix = matrices[0]

    for m in matrices[1:]:
        bbrd_matrix = gb.binary.first(bbrd_matrix | m)

    return build_adj_array(bbrd_matrix, k=n, enforce_symmetry=True)


def build_adj_array(mtrx: Matrix, k: int, enforce_symmetry: bool = True) -> np.ndarray:
    """
    Build adjacency array from bfs matrix.
    """
    # enforce symmetry in final index array:
    # - build first compact k-neighbor array of indices
    # - convert back array to sparse matrix
    # - remove non-symmetric edges
    # - convert back to compact array
    # - fill empty values
    #
    # NOTE: this could be simpler. If `ss.kselect` had support for "smallest"
    # we would skip the first two steps altogether. Another strategy would be
    # to build the array incrementally but with something like numba / cython
    khop_arr = mtrx.ss.compactify(how="smallest", k=k, asindex=True)

    khop_arr = khop_arr.to_dense(fill_value=-1, dtype="int32")

    if enforce_symmetry is False:
        return khop_arr

    khop_matrix = _gb_matrix_from_adj_arr(khop_arr)

    khop_matrix << khop_matrix.ewise_mult(khop_matrix.T, op="times")

    khop_fnl = khop_matrix.ss.compactify(how="first", k=k, asindex=True)

    return khop_fnl.to_dense(fill_value=-1, dtype="int32")


def khop_nneighbors(
    inc_matrix: csr_matrix | csc_matrix,
    inc_type: str = "v2e",
    n: int | None = None,
    k: int = 3,
    dilations: int = 1,
):
    """
    Find `n` neighbors after `k` hops given a sparse incidence matrix representing a graph.

    Parameters
    ----------
    inc_matrix : csr_matrix | csc_matrix
        An incidence matrix representing a graph.
    inc_type : {'v2e', 'e2v', 'e2f'}
        The type of input incidence matrix.
    k : int
        Number of hops / bfs levels to reach.
    n : int
        Number of neighbors to keep
    dilations : int
        Dilations of bfs hops.

    Returns
    -------
    np.ndarray
        An array of shape `(l, n)` where `l` the number of rows of the incidence matrix
        and `n` the number of neighbors.
    """
    bfs_matrix = get_bfs_matrix(
        inc_matrix=inc_matrix, inc_type=inc_type, k=k, dilations=dilations, _max=n
    )

    return build_adj_array(bfs_matrix, k=n, enforce_symmetry=True)


def get_global_matrix(
    nrows: int, n_global: int, rng: np.random.RandomState | None, sym: bool = False
) -> Matrix:
    """
    Get a matrix with unidirectional connections with `n_global` nodes.
    """
    if rng is None:
        rng = np.random.RandomState()

    cols = np.arange(n_global)
    # cols = rng.choice(nrows, n_global, replace=False)

    row_idx = np.repeat(np.arange(nrows), n_global)
    col_idx = np.tile(cols, nrows)

    glbl_matrix = Matrix.from_coo(
        row_idx, col_idx, -2, gb.dtypes.INT64, nrows=nrows, ncols=nrows
    )

    if sym is True:
        glbl_matrix << glbl_matrix.ewise_add(glbl_matrix.T, op="min")

    return glbl_matrix


def get_random_matrix(
    nrows: int, n_random: int, rng: np.random.RandomState | int, sym: bool = True
) -> Matrix:
    """
    Get a matrix with `n_random` connections per node between `nrows` nodes.
    """
    if rng is None:
        rng = np.random.RandomState()

    if sym is True:
        n_random = n_random // 2

    # Random connections
    vals = rng.choice(nrows**2, nrows * n_random, replace=False)

    col_idx = vals // nrows
    row_idx = vals % nrows

    rand_matrix = Matrix.from_coo(
        row_idx, col_idx, -1, gb.dtypes.INT64, nrows=nrows, ncols=nrows
    )

    if sym is True:
        rand_matrix << rand_matrix.ewise_add(rand_matrix.T, op="min")

    return rand_matrix


def get_bfs_matrix(
    inc_matrix: csr_matrix | csc_matrix,
    inc_type: str = "v2e",
    k: int = 3,
    dilations: int = 1,
    _max: int | None = None,
) -> Matrix:
    """
    Get a matrix of bfs levels up to `k` hops and `l` dilations

    Parameters
    ----------
    inc_matrix : csr_matrix | csc_matrix
        An incidence matrix representing a graph.
    inc_type : {'v2e', 'e2v', 'e2f'}
        The type of input incidence matrix.
    k : int
        Number of hops / bfs levels to reach.
    dilations : int
        Dilations of bfs hops.
    _max : int | None
        Number of maximum average neighbors. If provided, it acts as an early stop
        where the bfs op stops when the average collected neighbors reach this number.
    """
    inc_matrix = scipy_to_gb(inc_matrix)

    if inc_type in ("e2v", "f2e"):
        adj_matrix = inc_to_line_graph_adj_matrix(inc_matrix)

    elif inc_type in ("v2e", "e2f"):
        adj_matrix = inc_to_adj_matrix(inc_matrix)

    bfs_matrix = bfs_levels(
        adj_matrix, source=None, steps=k, dilations=dilations, _max=_max
    )

    # select dilations
    if dilations > 1:
        bfs_matrix << bfs_matrix.select(bfs_matrix % dilations == 0)

    # enforce symmetry
    bfs_matrix << bfs_matrix.ewise_add(bfs_matrix.T, op="min")

    return bfs_matrix


def init_bfs(source: int | ArrayLike | None, n: int) -> Tuple[Matrix, Matrix]:
    """
    Initialize utility distance (`D`) and frontier matrices (`F`) for bfs.

    Parameters
    ----------
    source : int | ArrayLike | None
        Vertices to start breadth first search operations from. Multiple vertices
        results in multiple simultaneous searches. If `None`, all vertices will
        be considered as source.
    n : int
        The number of vertices in the graph.

    Returns
    -------
    D : Matrix
        A sparse matrix representing the distances from starting vertices.
        Also used to track visited vertices.
    F : Matrix
        A boolean sparse matrix representing the frontier.
    """
    if source is None:
        D = gb.Vector.from_scalar(0, n, gb.dtypes.INT64).diag()

    else:
        source = [source] if isinstance(source, int) else source

        nrows = len(source)
        ncols = n

        col_idx = np.array(source)
        row_idx = np.arange(nrows, dtype=np.int64)

        D = Matrix.from_coo(
            row_idx, col_idx, 0, gb.dtypes.INT64, nrows=nrows, ncols=ncols
        )

    F = D.apply(gb.unary.one[gb.dtypes.BOOL]).new()

    return D, F


def bfs_levels(
    adj_matrix: Matrix,
    source: int | ArrayLike | None = None,
    steps: int | None = 3,
    dilations: int = 1,
    _max: int | None = None,
) -> np.ndarray:
    """
    Perform bfs levels.

    Adjusted from:
        https://github.com/python-graphblas/graphblas-algorithms/blob/main/graphblas_algorithms/algorithms/_bfs.py

    Parameters
    ----------
    adj_matrix : gb.Matrix
        The input adjacency matrix.
    source : int | ArrayLike | None
        Vertices to start breadth first search operations from. Multiple vertices
        results in multiple simultaneous searches. If `None`, all vertices will
        be considered as source.
    steps : int | None
        Number of bfs steps to take. If `None`, run until exhaustion. In the case
        that no source is provided this may result to a full `(n, n)` matrix.
    dilations : int
        Dilations of bfs hops.
    _max : int | None
        Number of maximum average neighbors. If provided, it acts as an early stop
        where the bfs op stops when the average collected neighbors reach this number.

    Returns
    -------
    neighbors : np.ndarray
        The neighbors.
    distances : np.ndarray
        The neighbor distances from starting nodes.
    """
    _max = float("inf") if _max is None else _max

    n = adj_matrix.nrows

    D, F = init_bfs(source, n)

    steps = n if (steps is None or steps > n) else steps + 1

    for i in range(1, steps):
        step = gb.semiring.any_pair[gb.dtypes.BOOL](F @ adj_matrix)

        F(~D.S, replace=True) << step

        if F.nvals == 0:
            break

        D(F.S) << i

        F = update_frontier(D, F, _max, dilations)

    return D


def update_frontier(D: Matrix, F: Matrix, _max: int | float, dilations: int) -> Matrix:
    """
    Update bfs frontier represented as a sparse matrix.

    D : Matrix
        A sparse matrix used to track visited vertices.
    F : Matrix
        A boolean sparse matrix representing the frontier.
    _max : int | None
        Number of maximum neighbors per row.
    dilations : int
        Dilations of bfs hops.

    Returns
    -------
    F : Matrix
        A boolean sparse matrix representing the frontier
    """
    # check bfs stop criterion per row and update frontier matrix `F`
    if dilations == 1:
        row_mask = (D.reduce_rowwise(gb.agg.count) < _max).diag()

    elif dilations > 1:
        row_mask = (
            D.select(D % dilations == 0).reduce_rowwise(gb.agg.count) < _max
        ).diag()

    # Remove zero entries from the matrix
    row_mask(row_mask.V, replace=True) << row_mask

    F << gb.semiring.lor_land[gb.dtypes.BOOL](F @ row_mask)

    return F


def _gb_matrix_from_adj_arr(adj_arr: np.ndarray, fill_value: int = -1) -> Matrix:
    """
    Create a `graphblas` Matrix from an adjacency array.
    """
    n, d = adj_arr.shape

    row_idx = np.repeat(np.arange(n), d)
    col_idx = adj_arr.flatten()

    mask = col_idx != fill_value

    row_idx = row_idx[mask]
    col_idx = col_idx[mask]

    return Matrix.from_coo(row_idx, col_idx, 1, gb.dtypes.INT64, nrows=n, ncols=n)


# NOTE: Testing functions
def gb_is_symmetric(mtrx):
    """
    Check if a graphblas matrix is symmetric.
    """
    tmp_matrix = mtrx.ewise_add(mtrx.T, op="ne").dup(gb.dtypes.INT32)

    return tmp_matrix.reduce_scalar(gb.agg.count_nonzero) == 0


def scp_is_symmetric(mtrx):
    """
    Check if a scipy matrix is symmetric.
    """
    return (mtrx != mtrx.transpose()).nnz == 0


def scp_sparse_from_adj_arr(adj_arr: np.ndarray, fill_value: int = -1) -> csr_matrix:
    """
    Create a `scipy` sparse Matrix from an adjacency array.
    """
    n, d = adj_arr.shape

    row_idx = np.repeat(np.arange(n), d)
    col_idx = adj_arr.flatten()

    mask = col_idx != fill_value

    row_idx = row_idx[mask]
    col_idx = col_idx[mask]

    data = np.ones(len(row_idx))

    A = csr_matrix((data, (row_idx, col_idx)), shape=(n, n))

    return A


def adj_arr_is_symmetric(mtrx, k=32):
    """
    Test whether a graph rerpresented by an adjacency array is symmetric.
    """
    khop = mtrx.ss.compactify(how="smallest", k=k, asindex=True)

    adj_arr = khop.to_dense(fill_value=-1, dtype="int32")

    A = scp_sparse_from_adj_arr(adj_arr)

    return (A != A.T).nnz
