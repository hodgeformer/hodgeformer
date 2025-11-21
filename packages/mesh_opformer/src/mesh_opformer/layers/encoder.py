from typing import Callable, Tuple, List

import re
import torch

from torch import nn

from .utils import clones
from .utils import NoOperation

from .other import (
    interweave_two_lists,
    _parse_layer_layout_string,
    _handle_layer_layout,
)


class MeshHodgeFormerEncoder(nn.Module):
    """
    The `MeshHodgeFormerEncoder` or Mesh Hodge Transformer is made up of stacked
    layers that consist of:

      * mesh multi-head attention based on hodge attention layers
      * feed-forward layers

    """

    def __init__(
        self,
        embed_layer: nn.Module,
        laplace_layer: nn.Module | None,
        dirac_layer: nn.Module | None,
        basic_layer: nn.Module | None,
        N: int,
        modes: Tuple[str] | List[str],
        layer_layout: str,
    ):
        """
        Parameters
        ----------
        embed_layer : nn.Module
            A module to embed input features.
        hodge_layer : nn.Module
            A module of instance of `MeshHodgeFormerLayer` class.
        basic_layer : nn.Module | None
            A module of instance of `VanillaTransformerLayer` class.
        N : int
            Number of layers.
        modes : Tuple[str]
            Mesh elements 'v', 'e' or 'f' on which the layer will operate.
        layer_layout : str
            How the hodge laplace, hodge dirac and basic layers are interleaved.
            Expected input is a nested formatted string of the form:

                                "\(\d+:\d+e?\):\d+e?"

            The string defines two ratios. One betweemn the hodge laplace and hodge
            dirac layers, and one between these and the basic layers.

            For example:
            - (2:1e):1e means 2 hodge laplace layers, followed by 1 dirac layer
            followed by 1 basic layer.
            - (4:1e):1e means 4 hodge laplace layers, followed by 1 dirac layer
            followed by 1 basic layer,
        """
        super(MeshHodgeFormerEncoder, self).__init__()

        self.layer_layout = _parse_layer_layout_string(layer_layout)

        self.N_h, self.N_b = _handle_layer_layout(N, self.layer_layout)

        self.embed_layer = embed_layer

        if laplace_layer is not None:
            self.laplace_layers = clones(laplace_layer, self.N_h)

        # if dirac_layer is not None:
        #     self.dirac_layers = clones(dirac_layer, self.N_d)

        # Initialize norm layers
        self.v_norm = nn.LayerNorm(laplace_layer.d_v) if "v" in modes else nn.Identity()
        self.e_norm = nn.LayerNorm(laplace_layer.d_e) if "e" in modes else nn.Identity()
        self.f_norm = nn.LayerNorm(laplace_layer.d_f) if "f" in modes else nn.Identity()

        # Add vanilla transformer layers
        b_no_ops = clones(NoOperation(), self.N_b)

        if basic_layer is None:
            self.v_basic = b_no_ops
            self.e_basic = b_no_ops
            self.f_basic = b_no_ops

        else:
            self.v_basic = clones(basic_layer, self.N_b) if "v" in modes else b_no_ops
            self.e_basic = clones(basic_layer, self.N_b) if "e" in modes else b_no_ops
            self.f_basic = clones(basic_layer, self.N_b) if "f" in modes else b_no_ops

    def forward(self, x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx, *args, **kwargs):
        """
        Pass the input (and mask) through each layer in turn.

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
        v_idx : torch.Tensor
            Indices indicating a local vertex neighborhood as tensor of shape (n, k_v).
        e_idx : torch.Tensor
            Indices indicating a local edge neighborhood as tensor of shape (m, k_e).
        f_idx : torch.Tensor
            Indices indicating a local face neighborhood as tensor of shape (k, k_f).
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
        x_v, x_e, x_f = self.embed_layer(x_v, x_e, x_f, d_0, d_1)

        l_layers = [("L", i) for i in range(self.N_h)]
        d_layers = [("D", i) for i in range(self.N_b)]

        h, d, _ = self.layer_layout

        layers = interweave_two_lists(l_layers, d_layers, h, d)

        for layer_type, idx in layers:

            if layer_type == "L":
                layer = self.laplace_layers[idx]

                x_v, x_e, x_f = layer(
                    x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx, *args, **kwargs
                )

            elif layer_type == "D":
                # layer = self.dirac_layers[idx]

                # x_v, x_e, x_f = layer(
                #     x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx, *args, **kwargs
                # )

                x_v = self.v_basic[idx](x_v, idx=v_idx, mask=None)
                x_e = self.e_basic[idx](x_e, idx=e_idx, mask=None)
                x_f = self.f_basic[idx](x_f, idx=f_idx, mask=None)

        return self.v_norm(x_v), self.e_norm(x_e), self.f_norm(x_f)


class MeshHodgeFormerLayer(nn.Module):
    """
    The basic layer of the `MeshHodgeFormerLayer` or Mesh Hodge Transformer
    architecture is a combination of (1) hodge-attention and (2) feed
    forward layers connected with residual connections.
    """

    def __init__(self, d_v, d_e, d_f, mesh_attn, v_ff, e_ff, f_ff, dropout, modes):
        """
        Parameters
        ----------
        v_size : int
            The vertices output size.
        e_size : int
            The edges output size.
        mesh_attn : nn.Module
            A self-attention module.
        feed_forward : nn.Module
            A feed forward module.
        dropout : float
            Dropout probability during training.
        modes : Tuple[str]
            The mesh elements 'v', 'e' or 'f' on which attention operations
            will be defined.
        """
        super(MeshHodgeFormerLayer, self).__init__()

        self.mesh_attn = mesh_attn

        self.v_ff = v_ff
        self.e_ff = e_ff
        self.f_ff = f_ff

        self.norm_res_triple = SublayerResidualTripleInput(
            d_v, d_e, d_f, dropout, modes
        )

        self.v_norm = (
            SublayerResidualSingleInput(d_v, dropout) if "v" in modes else NoOperation()
        )
        self.e_norm = (
            SublayerResidualSingleInput(d_e, dropout) if "e" in modes else NoOperation()
        )
        self.f_norm = (
            SublayerResidualSingleInput(d_f, dropout) if "f" in modes else NoOperation()
        )

        self.d_v = d_v
        self.d_e = d_e
        self.d_f = d_f

    def forward(self, x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx, mask):
        """
        Applies a `MeshOpFormer` layer to the input tensors.

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
        v_idx : torch.Tensor
            Indices indicating a local vertex neighborhood as tensor of shape (v, k_v).
        e_idx : torch.Tensor
            Indices indicating a local edge neighborhood as tensor of shape (v, k_e).
        f_idx : torch.Tensor
            Indices indicating a local face neighborhood as tensor of shape (v, k_f).
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
        x_v, x_e, x_f = self.norm_res_triple(
            x_v,
            x_e,
            x_f,
            d_0,
            d_1,
            v_idx,
            e_idx,
            f_idx,
            lambda x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx: self.mesh_attn(
                x_v, x_e, x_f, d_0, d_1, v_idx, e_idx, f_idx, mask
            ),
        )

        x_v = self.v_norm(x_v, self.v_ff)
        x_e = self.e_norm(x_e, self.e_ff)
        x_f = self.f_norm(x_f, self.f_ff)

        return x_v, x_e, x_f


# Abstractions for residual+norm connections
class SublayerResidualSingleInput(nn.Module):
    """
    This class represents a module that implements residual
    connection along with Layer Normalization.

    NOTE: The layer follows the so-called pre-Normalization
    convention, where normalization is performed before the
    input features go into the sublayer.

    More here:

    `On Layer Normalization in the Transformer Architecture`
    `https://arxiv.org/pdf/2002.04745`
    """

    def __init__(self, size: int, dropout: float) -> None:
        """
        Parameters
        ----------
        size : int
            The layer size to apply the LayerNorm.
        dropout : float
            Dropout probability.
        """
        super(SublayerResidualSingleInput, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable, *args) -> torch.Tensor:
        """
        Applies residual connection to any sublayer with given size.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        sublayer : Callable
            A callable that receives and outputs a torch tensor.
        """
        return x + self.dropout(sublayer(self.norm(x), *args))


class SublayerResidualTripleInput(nn.Module):
    """
    This class represents a module that implements residual
    connetion along with Layer Normalization.

    NOTE: The layer follows the so-called pre-Normalization
    convention, where normalization is performed before the
    input features go into the sublayer.

    More here:

    `On Layer Normalization in the Transformer Architecture`
    `https://arxiv.org/pdf/2002.04745`
    """

    def __init__(
        self, d_v: int, d_e: int, d_f: int, dropout: float, modes: Tuple[str]
    ) -> None:
        """
        Parameters
        ----------
        d_v : int
            Dimensionality of vertex embeddings.
        d_e : int
            Dimensionality of edge embeddings.
        d_f : int
            Dimensionality of face embeddings.
        dropout : float
            Dropout probability.
        modes : Tuple[str]
            The mesh elements 'v', 'e' or 'f' on which attention operations
            will be defined.
        """
        super(SublayerResidualTripleInput, self).__init__()

        self.v_norm = nn.LayerNorm(d_v) if "v" in modes else nn.Identity()
        self.e_norm = nn.LayerNorm(d_e) if "e" in modes else nn.Identity()
        self.f_norm = nn.LayerNorm(d_f) if "f" in modes else nn.Identity()

        self.v_dropout = nn.Dropout(dropout) if "v" in modes else nn.Identity()
        self.e_dropout = nn.Dropout(dropout) if "e" in modes else nn.Identity()
        self.f_dropout = nn.Dropout(dropout) if "f" in modes else nn.Identity()

    def forward(
        self,
        x_v: torch.Tensor,
        x_e: torch.Tensor,
        x_f: torch.Tensor,
        d_0: torch.Tensor,
        d_1: torch.Tensor,
        v_idx: torch.Tensor,
        e_idx: torch.Tensor,
        f_idx: torch.Tensor,
        sublayer: Callable,
    ) -> torch.Tensor:
        """
        Applies residual connection to any sublayer with given size.

        Parameters
        ----------
        x_v : torch.Tensor
            Node features as tensor of shape (n, d_v).
        x_e : torch.Tensor
            Edge features as tensor of shape (m, d_e).
        x_f : torch.Tensor
            Face features as tensor of shape (k, d_f).
        d_0 : torch.Tensor
            Incidence matrix (m, n) as a boundary operator from edges to vertices.
        d_1 : torch.Tensor
            Incidence matrix (k, m) as a boundary operator from faces to edges.
        v_idx : torch.Tensor
            Indices indicating a local vertex neighborhood as tensor of shape (n, k_v).
        e_idx : torch.Tensor
            Indices indicating a local edge neighborhood as tensor of shape (m, k_e).
        f_idx : torch.Tensor
            Indices indicating a local face neighborhood as tensor of shape (k, k_f).
        sublayer : Callable
            A callable that receives and outputs a torch tensor.
        """
        _x_v = self.v_norm(x_v)
        _x_e = self.e_norm(x_e)
        _x_f = self.f_norm(x_f)

        _x_v, _x_e, _x_f = sublayer(_x_v, _x_e, _x_f, d_0, d_1, v_idx, e_idx, f_idx)

        return (
            x_v + self.v_dropout(_x_v),
            x_e + self.e_dropout(_x_e),
            x_f + self.f_dropout(_x_f),
        )
