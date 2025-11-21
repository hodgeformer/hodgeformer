import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("`matplotlib` is required for some plotting functions.")


def frobenius_norm(M):
    """
    Calculate matrix frobenius norm
    """
    return np.linalg.norm(M, ord="fro")


def shrinkage_operator(arr, tau):
    """
    Shrinkage operator applied element-wise to n-dimensional arrays.

                S(x, tau) = sgn(x) * max(|x| - tau, 0),


    Parameters
    ----------
    arr : ArrayLike
        A numpy array to apply shrinkage on.
    tau : ArrayLike
        Shrinkage value to apply. Must be broadcastable with `arr`,
        i.e. a scalar or an array of same shape with `arr`.

    Returns
    -------
    ArrayLike
        Numpy array with its values thresholded by `tau`.
    """
    # Inplace implementation
    # _arr = np.abs(arr) - tau
    # np.maximum(_arr, 0, out=_arr)
    # np.multiply(np.sign(arr), _arr, out=_arr)
    #
    # return _arr
    _arr = np.abs(arr) - tau

    return np.sign(arr) * np.maximum(_arr, 0)


def singular_value_thresholding_operator(a, tau):
    """
    Apply the shrinkage operator to the singular values of a matrix.

    Parameters
    ----------
    a : ArrayLike
        A 2D array to apply the operator on.
    tau : float
        Threshold value to compare singular values with.

    Returns
    -------
    ArrayLike
        A 2D array on which singular values have been thresholded.
    """
    # Splitting a matrix (m, n) into matrices with shapes:
    #   ((m, n), (n, n), (n, n))
    U, s, Vh = np.linalg.svd(a, full_matrices=False)

    # Apply shrink operator to vector with singular values `s`
    _s = shrinkage_operator(s, tau)

    # Similar to U @ np.diag(_s) @ Vh
    return (U * _s) @ Vh


class RobustPCA:
    """
    A class to to represent a `RobustPCA` operator.

    Robust Principal Component Analysis attempts to recover a sparse
    signal applied as corruption or perturbation on a low-rank data
    matrix via iterative `Principal Component Pursuit`. Among all
    feasible decompositions, a weighted combination of the nuclear
    norm and the `l1` norm is identified.

                        minimize |L|. + λ*|S|1
                        subject to L + S = M

    where |o|. is the nuclear norm and |o|1 the l1 norm. The above
    optimization problem is handled in the paper with ALM (Augmented
    Lagrange Multipliers):

    `Robust Principal Component Analysis`
    `https://arxiv.org/pdf/0912.3599.pdf`

    """

    def __init__(self, D, mu=None, lmbda=None):
        """
        Initializer for a `RobustPCA` instance.

        Parameters
        ----------
        D : ArrayLike
            A 2D
        mu : float

        lmbda : float
        """
        self.D = D

        if mu:
            self.mu = mu
        else:
            self.mu = self.D.size / (4 * frobenius_norm(self.D))

        self.mu_inv = 1.0 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1.0 / np.sqrt(np.max(self.D.shape))

    def fit(self, tol=1e-2, max_iter=1000, verbose=0):
        """
        Fit a `RobustPCA` instance.


        Parameters
        ----------
        tol : float | None
            Convergence tolerance to end the optimization process.
        max_iter : int
            Maximum number of iterations.
        verbose : int
            Verbosity level for the fit procedure.
        """
        err = np.Inf

        Lk = np.zeros_like(self.D, dtype="float32")
        Sk = np.zeros_like(self.D, dtype="float32")
        Yk = np.zeros_like(self.D, dtype="float32")

        if tol is None:
            tol = 1e-7 * frobenius_norm(self.D)

        i = 1

        while (err > tol) and i < max_iter:

            # D - Sk - Yk * 1/mu
            Lk = singular_value_thresholding_operator(
                self.D - Sk + (Yk * self.mu_inv), self.mu_inv
            )

            # D - Lk - Yk * 1/mu
            Sk = shrinkage_operator(
                self.D - Lk + (Yk * self.mu_inv), self.mu_inv * self.lmbda
            )

            Yk = Yk + self.mu * (self.D - Lk - Sk)

            err = frobenius_norm(self.D - Lk - Sk)

            i += 1

            if verbose > 0:

                if (i % 100) == 0 or i > max_iter or err <= tol:
                    print("iteration: {0}, error: {1}".format(i, err))

        self.L = Lk
        self.S = Sk

        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):
        """
        Not clear what is happening.
        """

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)

        print("ymin: {0}, ymax: {1}".format(ymin, ymax))

        numplots = min(n, nrows * ncols)

        fig = plt.figure(figsize=(7, 8))

        for n in range(numplots):

            ax = fig.add_subplot(nrows, ncols, n + 1)

            ax.set_ylim((ymin - tol, ymax + tol))
            ax.plot(self.L[n, :] + self.S[n, :], "r")
            ax.plot(self.L[n, :], "b")

            if not axis_on:
                ax.set_axis_off()

        return fig


def robust_pca(X, mu=None, lambd=None, tol=None, max_iter=1000, verbose=0):
    """
    Functional version of the `RobustPCA` class.

    Parameters
    ----------
    tol : float
        Convergence tolerance to end the optimization process.
    max_iter : int
        Maximum number of iterations.
    """
    if lambd is None:
        lambd = 1 / np.sqrt(np.max(np.shape(X)))

    if mu is None:
        mu = 10 * lambd

    if mu is None:
        mu = X.size / (4 * frobenius_norm(X))

    if lmbda is None:
        lmbda = 1.0 / np.sqrt(np.max(X.shape))

    if tol is None:
        tol = 1e-7 * frobenius_norm(X)

    mu_inv = 1.0 / mu

    Yk = np.zeros_like(X)
    Lk = np.zeros_like(X)
    Sk = np.zeros_like(X)

    for i in range(max_iter):

        # ADMM step: update L and S

        # D - Sk - Yk * 1/mu
        Lk = singular_value_thresholding_operator(X - Sk + (Yk * mu_inv), mu_inv)

        # D - Lk - Yk * 1/mu
        Sk = shrinkage_operator(X - Lk + (Yk * mu_inv), mu_inv * lmbda)

        # and augmented lagrangian multiplier
        Z = X - Lk - Sk
        Yk = Yk + mu * Z

        err = np.linalg.norm(Z) / np.linalg.norm(X)

        if verbose > 0:

            if (i % 100) == 0 or i > max_iter or err <= tol:
                print("iteration: {0}, error: {1}".format(iter, err))

        if err <= tol:
            break

    return Lk, Sk


def test_robust_pca() -> None:
    """
    Default example test for `robust_pca`.
    """
    max_iter = 100
    mu = 0.4

    M = np.random.rand(500, 60)

    L, S = robust_pca(M, max_iter=max_iter, mu=mu)

    ff0 = np.linalg.matrix_rank(M)
    ff1 = np.linalg.matrix_rank(L)
    ff2 = np.linalg.matrix_rank(S)

    RR = (np.linalg.norm(S, axis=1)) - 0.5 > (np.linalg.norm(L, axis=1))

    des = np.count_nonzero(RR)

    print("ok")
