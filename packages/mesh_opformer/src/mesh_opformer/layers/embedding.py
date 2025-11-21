from typing import Tuple

import torch

from torch import nn

from .fc import MLPTwoLayer, MLPSimple


class LinearEmbedding(nn.Module):
    """
    A class to represent a linear embedding for mesh elements.

    For each input mesh element we apply a learnable linear transformation.
    """

    def __init__(
        self, v_in: int, e_in: int, f_in: int, d_v: int, d_e: int, d_f: int
    ) -> None:
        """
        Initializer for the `MeshInputEmbedding` class.

        Parameters
        ----------
        v_in : int
            Input dimensionality of vertex features.
        e_in : int
            Input dimensionality of edge features.
        f_in : int
            Input dimensionality of face features.
        d_v : int
            Dimensionality of vertex embeddings.
        d_e : int
            Dimensionality of edge embeddings.
        d_f : int
            Dimensionality of face embeddings.
        """
        super(LinearEmbedding, self).__init__()

        self.v_embed = MLPTwoLayer(v_in, d_v * 2, d_v)
        self.e_embed = MLPTwoLayer(e_in, d_e * 2, d_e)
        self.f_embed = MLPTwoLayer(f_in, d_f * 2, d_f)

    def forward(
        self, x_v: torch.Tensor, x_e: torch.Tensor, x_f: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward step.
        """
        x_v = self.v_embed(x_v)
        x_e = self.e_embed(x_e)
        x_f = self.f_embed(x_f)

        return x_v, x_e, x_f


class NeighborEmbedding(nn.Module):
    """
    A class to represent an embedding layer for mesh elements.

    For each input mesh element we aggregate neighboring element
    features.
    """

    def __init__(
        self, n: int, v_in: int, e_in: int, f_in: int, d_v: int, d_e: int, d_f: int
    ) -> None:
        """
        Initializer for the `NeighborEmbedding` class.

        Parameters
        ----------
        v_in : int
            Input dimensionality of vertex features.
        e_in : int
            Input dimensionality of edge features.
        f_in : int
            Input dimensionality of face features.
        d_v : int
            Dimensionality of vertex embeddings.
        d_e : int
            Dimensionality of edge embeddings.
        d_f : int
            Dimensionality of face embeddings.
        """
        super(NeighborEmbedding, self).__init__()

        assert d_v % 2 == 0 and d_e % 2 == 0 and d_f % 2 == 0

        d_v_h = d_v
        d_e_h = d_e
        d_f_h = d_f

        self.v_embed = MLPSimple([v_in] + [d_v] * n + [d_v_h])
        self.e_embed = MLPSimple([e_in] + [d_e] * n + [d_e_h])
        self.f_embed = MLPSimple([f_in] + [d_f] * n + [d_f_h])

    def forward(
        self,
        x_v: torch.Tensor,
        x_e: torch.Tensor,
        x_f: torch.Tensor,
        d_0: torch.Tensor,
        d_1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embeds input tensors.

        Parameters
        ----------
        x_v : torch.Tensor
            Node features as tensor of shape (v, d_v).
        x_e : torch.Tensor
            Edge features as tensor of shape (e, d_e).
        x_f : torch.Tensor
            Edge features as tensor of shape (f, d_f).
        d_0 : torch.Tensor
            Incidence matrix (m, n) as a boundary operator taking vertices to edges.
        d_1 : torch.Tensor
            Incidence matrix (k, m) as a boundary operator taking edges to faces.
        mask : torch.Tensor
            A mask to be applied on the attention matrix.

        Returns
        -------
        x_v : torch.Tensor
            Node embeddings.
        x_e : torch.Tensor
            Edge embeddings.
        x_f : torch.Tensor
            Edge embeddings.
        """
        x_v_dual = apply_adjacency(x_v, d_0, normalize=True, renorm=True)
        x_e_dual = apply_adjacency(x_e, d_1, normalize=True, renorm=True)

        x_v = self.v_embed(x_v_dual)
        x_e = self.e_embed(x_e_dual)
        x_f = self.f_embed(x_f)

        return x_v, x_e, x_f


def build_deg_matrix_from_incidence(d: torch.Tensor):
    """
    Build the degree matrix from a boundary operator.
    """
    return torch.abs(d).sum(dim=1).to_dense().unsqueeze(-1)


def apply_laplacian(
    x: torch.Tensor, d: torch.Tensor, normalize: bool = False, renorm: bool = False
) -> torch.Tensor:
    """
    Aggregate features via the (normalized) laplacian.

    Parameters
    ----------
    x : torch.Tensor
        Features as tensor of shape (n, d) where n the number
        of vertices or edges and the input dimensionality.
    d : torch.Tensor
        Incidence matrix `(m, n)` as a boundary operator taking e.g.
        `n` vertices or edges to `m` edges or faces respectively.
    normalized : bool
        Whether to normalize the laplacian.

    Returns
    -------
    torch.Tensor
        The neighborhood aggregated features.
    """
    _x = x

    _deg = build_deg_matrix_from_incidence(d)

    if normalize is True:
        deg = torch.add(_deg, 1.0) if renorm is True else _deg
        deg_inv = torch.reciprocal(deg.sqrt())

    if normalize is True:
        x = torch.mul(deg_inv, x)

    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=False):
        x = x.float()
        x = torch.bmm(d.transpose(-2, -1), torch.bmm(d, x))

    # Apply D_inv
    if normalize is True:
        x = torch.mul(deg_inv, x)

    return _x + x


def apply_adjacency(
    x: torch.Tensor, d: torch.Tensor, normalize: bool = False, renorm: bool = False
) -> torch.Tensor:
    """
    Aggregate features via the Adjacency matrix.

    Parameters
    ----------
    x : torch.Tensor
        Features as tensor of shape (n, d) where n the number
        of vertices or edges and the input dimensionality.
    d : torch.Tensor
        Incidence matrix `(m, n)` as a boundary operator taking e.g.
        `n` vertices or edges to `m` edges or faces respectively.

    Returns
    -------
    torch.Tensor
        The neighborhood aggregated features.
    """
    _x = x

    _deg = build_deg_matrix_from_incidence(d)

    if normalize is True:
        deg = torch.add(_deg, 1.0) if renorm is True else _deg
        deg_inv = torch.reciprocal(deg.sqrt())

    if normalize is True:
        x = torch.mul(deg_inv, x)

    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=False):
        x = x.float()
        x = torch.mul(_deg, x) - torch.bmm(d.transpose(-2, -1), torch.bmm(d, x))

    if normalize is True:
        x = torch.mul(deg_inv, x)

    return _x + x
