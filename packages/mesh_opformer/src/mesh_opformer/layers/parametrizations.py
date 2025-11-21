import torch

from torch import nn


class HodgeDiagonal(nn.Module):
    """
    Class to represent a diagonal Hodge Matrix.
    """


class HodgeFull(nn.Module):
    """
    Class to represent a full Hodge matrix.
    """


def degree_matrix(
    h_0: torch.Tensor, h_1: torch.Tensor, d_0: torch.Tensor
) -> torch.Tensor:
    """
    Calculate degree matrix via element wise multiplications.

    This implementation is non-differentiable.

    Parameters
    ----------
    h_0 : torch.Tensor
        Diagonal hodge operator for nodes.
    h_1 : torch.Tensor
        Diagonal hodge operator for edges.
    d_0 : torch.sparse_coo
        Boundary operator.

    Returns
    -------
    torch.Tensor
        The graph degree tensor.
    """
    h = h_0.size(1)

    h_1 = h_1.detach()
    h_0 = h_0.detach()

    degs = torch.stack(
        [
            torch.abs(
                h_0[:, i, ...].unsqueeze(2)
                * d_0.transpose(-2, -1)
                * h_1[:, i, ...].unsqueeze(1)
            )
            .sum(dim=-1)
            .to_dense()
            for i in range(h)
        ],
        dim=1,
    ).sqrt()

    degs = torch.clamp(degs, 1e-6)

    degs = torch.reciprocal(degs)

    return torch.nan_to_num(degs, 1.0)


def degree_matrix_2(
    h_0: torch.Tensor, h_1: torch.Tensor, d_0: torch.Tensor
) -> torch.Tensor:
    """
    Calculate degree matrix via element wise multiplications.

    This implementation is non-differentiable.

    Parameters
    ----------
    h_0 : torch.Tensor
        Diagonal hodge operator for nodes.
    h_1 : torch.Tensor
        Diagonal hodge operator for edges.
    d_0 : torch.sparse_coo
        Boundary operator.

    Returns
    -------
    torch.Tensor
        The graph degree tensor.
    """
    h = h_0.size(1)

    h_0 = torch.diag_embed(h_0)
    h_1 = torch.diag_embed(h_1)

    h_1 = h_1.detach()
    h_0 = h_0.detach()

    degs = torch.stack(
        [torch.bmm(torch.abs(d_0).transpose(-2, -1), h_1[:, i, ...]) for i in range(h)],
        dim=1,
    )

    degs = torch.einsum("bhnn,bhnm->bhnm", h_0, degs)

    degs = degs.sum(dim=-1).sqrt()

    degs = torch.clamp(degs, 1e-6)

    degs = torch.reciprocal(degs)

    return torch.nan_to_num(degs, 1.0)
