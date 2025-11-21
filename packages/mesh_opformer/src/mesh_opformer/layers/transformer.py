from typing import Dict

import torch
import copy
import math

from torch import nn

from .encoder import SublayerResidualSingleInput
from .utils import clones
from .norm import L1Norm
from .fc import MLPTwoLayer


from .attention import (
    linear_attention__efficient,
    linear_attention__flowformer,
    linear_attention__fast_transformers,
    sparse_attention,
)

LINEAR_ATTENTION = {
    "full-linear": linear_attention__flowformer,
    "full-linear-efficient": linear_attention__efficient,
    "full-linear-flowformer": linear_attention__flowformer,
    "full-linear-fast-transformers": linear_attention__fast_transformers,
}


def init_layer__transformer(d: int, d_hidden: int, dropout: float, attn_kw: Dict):
    """
    Initialize a `Transformer` layer.
    """

    multi_head_attention = MultiHeadAttention(d=d, dropout=dropout, **attn_kw)

    mlp_kw = {"batch_norm": True, "activation": "relu", "dropout": dropout}

    ff = MLPTwoLayer(d, d_hidden, None, **mlp_kw)

    transformer_module = VanillaTransformerLayer(
        d,
        attn=copy.deepcopy(multi_head_attention),
        ff=copy.deepcopy(ff),
        dropout=dropout,
    )

    return transformer_module


class VanillaTransformerLayer(nn.Module):
    """
    The basic layer of a vanilla Transformer. It includes a multi-head
    attention layer and feed forward layers connected with residual
    connections.
    """

    def __init__(self, d, attn, ff, dropout):
        """
        Parameters
        ----------
        d : int
            The vertices output size.
        attn : nn.Module
            A self-attention module.
        feed_forward : nn.Module
            A feed forward module.
        dropout : float
            Dropout probability during training.
        """
        super(VanillaTransformerLayer, self).__init__()

        self.attn = attn

        self.ff = ff

        self.norm_attn = SublayerResidualSingleInput(d, dropout)
        self.norm_mlps = SublayerResidualSingleInput(d, dropout)
        self.d = d

    def forward(self, x, idx, mask):
        """
        Applies a `MeshOpFormer` layer to the input tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input features as tensor of shape (n, d).
        idx : torch.Tensor | None
            Indices indicating a local neighborhood as tensor of shape (n, k).
        mask : torch.Tensor
            A mask to be applied on the attention matrix.

        Returns
        -------
        x : torch.Tensor
            Output tensor of shape (n, d)
        """
        x = self.norm_attn(x, lambda x, idx: self.attn(x, idx, mask), idx)

        x = self.norm_mlps(x, self.ff)

        return x


class MultiHeadAttention(nn.Module):
    """
    A class representing a module with multiple attention heads.
    """

    def __init__(
        self,
        h: int,
        d: int,
        dropout: float = 0.1,
        dropout_attn: float | None = None,
        norm_type: str | bool = "layer_norm",
        attn_type: str = "grouped",
        **kwargs,
    ):
        """
        Take in model size and number of heads.

        Parameters
        ----------
        h : int
            Number of heads
        d : int
            The vertex embeddings dimension.
        dropout : float
            Dropout probability.
        attn_sym : bool
            Whether the calculated hodge operators will be symmetric or not.
        norm_type : {'layer_norm', 'rms_norm'} or False
            Type of normalization to apply to Q, K matrices.
        attn_type : {'full-linear', 'grouped'}
            Whether to apply full or sparse attention.
        """
        super(MultiHeadAttention, self).__init__()

        assert d % h == 0

        self.h = h

        self.d_k = d_k = d // h

        self.norm_type = norm_type
        self.attn_type = attn_type

        # Initialize linear mappings
        self.linears = self._initialize_linear_layers(d)

        norm_Q, norm_K = self._initialize_norm_layers(d_k, norm_type=norm_type)

        self.norm_Qs = clones(norm_Q, self.h)
        self.norm_Ks = clones(norm_K, self.h)

        # Initialize output linear layers
        self.linear_out = nn.Linear(d, d)

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_attn = dropout_attn

    def _initialize_linear_layers(self, d):
        """
        Initialize linear layers based on `modes` and `sym` arguments.
        """
        kw = {"bias": False}

        return nn.ModuleList([nn.Linear(d, d, **kw) for _ in range(3)])

    def _initialize_norm_layers(self, d, norm_type):
        """
        Initialize norm layers based on `modes` and `attn_sym` arguments.
        """
        if norm_type and norm_type not in ("layer_norm", "rms_norm", "l1_norm"):
            raise ValueError(
                (
                    "Given `norm_type` argument is not valid. "
                    "Choose from ('layer_norm', 'rms_norm', 'l1_norm')."
                )
            )

        layer_cls = {
            False: nn.Identity,
            "layer_norm": nn.LayerNorm,
            "rms_norm": nn.RMSNorm,
            "l1_norm": L1Norm,
        }
        layer_kws = {
            "layer_norm": {"elementwise_affine": False, "bias": False},
        }

        norm_Q = layer_cls[norm_type](d, **layer_kws.get(norm_type, {}))
        norm_K = layer_cls[norm_type](d, **layer_kws.get(norm_type, {}))

        return norm_Q, norm_K

    def _apply_norm_across_heads(self, norms, x):
        """
        Apply each `LayerNorm` across heads
        """
        return torch.stack(
            [
                norm(_x)
                for norm, _x in zip(norms, (x[:, i, ...] for i in range(self.h)))
            ],
            dim=1,
        )

    def forward(self, x, idx=None, mask=None):
        """
        Forward step of multi head attention.

        Parameters
        ----------
        x_v : torch.Tensor
            Features as tensor of shape (n, d).
        idx : torch.Tensor | None
            Indices indicating a local neighborhood as tensor of shape (n, k).
        mask : torch.Tensor | None
            A mask tensor. Not supported for now.

        Returns
        -------
        x : torch.Tensor
            Output tensor of shape (n, d).
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        nbatches = x.size(0)

        # Calculate `q, k, v` linear projections
        Q, K, V = [
            linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for linear in self.linears
        ]

        if self.norm_type:
            Q = self._apply_norm_across_heads(self.norm_Qs, Q)
            K = self._apply_norm_across_heads(self.norm_Ks, K)

        # Apply attention on projected vectors in batch & concat
        # Apply mesh attention on vertices

        if self.attn_type == "grouped":
            x = sparse_attention(V, Q, K, idx, dropout=self.dropout_attn)

        elif self.attn_type in LINEAR_ATTENTION:
            x = LINEAR_ATTENTION[self.attn_type](v=V, q=Q, k=K)

            if self.dropout_attn is not None:
                x = nn.functional.dropout(x, p=self.dropout_attn)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        del Q
        del K
        del V

        return self.linear_out(x)
