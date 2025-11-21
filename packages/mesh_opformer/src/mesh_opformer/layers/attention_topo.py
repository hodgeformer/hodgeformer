from typing import Callable

import math
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import AccelerateOperator
from .utils import clones

from .parametrizations import degree_matrix

from .attention import (
    linear_attention__flowformer,
    linear_attention__fast_transformers,
    linear_attention__efficient,
)


class MeshMultiHeadAttention(nn.Module):
    """
    A class representing a module with multiple attention heads.


    - Q, K, V matrices to be calculated based on mode:

    |   | Q_v | K_v | V_v | Q_e | K_e | V_e | Q_t | K_t | V_t |
    |---|-----|-----|-----|-----|-----|-----|-----|-----|-----|
    | v | 1   | 1   | 1   | 1   | 1   |     |     |     |     |
    | e | 1   | 1   |     | 1   | 1   | 1   | 1   | 1   |     |
    | t |     |     |     | 1   | 1   |     | 1   | 1   | 1   |

    - if symmetric:

    |   | Q_v | K_v | V_v | Q_e | K_e | V_e | Q_t | K_t | V_t |
    |---|-----|-----|-----|-----|-----|-----|-----|-----|-----|
    | v | 1   |     | 1   | 1   |     |     |     |     |     |
    | e | 1   |     |     | 1   |     | 1   | 1   |     |     |
    | t |     |     |     | 1   |     |     | 1   |     | 1   |

    As no explicit inverse will be calculated we are going to make
    the following convention:

            *h = Q_h @ K_h.T    and    *h^(-1) = 1/c * K_h @ Q_h.T

    If `symmetric` is true:

            *h = Q_h @ Q_h.T    and    *h^(-1) = 1/c * Q_h @ Q_h.T

    NOTE
    Questions to be answered?

    1. Where and what kind of normalizations are required?

    2. Need to test the L0 @ x_v__V under what conditions is valid.
    What are the theoretical guarantess for this to work, in case
    no spectral decomposition takes place?

    3. `d_0` and `d_1` are sparse incident non-learnable matrices.
    Need to incorporate sparse - dense matrix computations. Cannot
    use `einsum` as it does not support sparse - dense computations.

    """

    LINEARS = {
        "v": [[1, 1, 1], [1, 1, 0], [0, 0, 0]],
        "e": [[1, 1, 0], [1, 1, 1], [1, 1, 0]],
        "f": [[0, 0, 0], [1, 1, 0], [1, 1, 1]],
    }

    LINEARS_SYM = {
        "v": [[1, 0, 1], [1, 0, 0], [0, 0, 0]],
        "e": [[1, 0, 0], [1, 0, 1], [1, 0, 0]],
        "f": [[0, 0, 0], [1, 0, 0], [1, 0, 1]],
    }

    NORMS = {
        "v": [[1, 1], [1, 1], [0, 0]],
        "e": [[1, 1], [1, 1], [1, 1]],
        "f": [[0, 0], [1, 1], [1, 1]],
    }

    NORMS_SYM = {
        "v": [[1, 0], [1, 0], [0, 0]],
        "e": [[1, 0], [1, 0], [1, 0]],
        "f": [[0, 0], [1, 0], [1, 0]],
    }

    def __init__(
        self,
        h,
        d_v,
        d_e,
        d_f,
        dropout=0.1,
        dropout_attn=None,
        attn_sym=True,
        attn_mask=False,
        norm_type="layer_norm",
        attn_acc=True,
        modes=("v", "e"),
    ):
        """
        Take in model size and number of heads.

        Parameters
        ----------
        h : int
            Number of heads
        d_v : int
            The vertex embeddings dimension.
        d_e : int
            The edge embeddings dimension.
        d_f : int
            The face embeddings dimension.
        dropout : float
            Dropout probability.
        attn_sym : bool
            Whether the calculated hodge operators will be symmetric or not.
        norm_type : {'layer_norm', 'rms_norm'} or False
            Type of normalization to apply to Q, K matrices.
        attn_acc : bool
            Whether to use accelerator operator.
        modes : Tuple[str]
            Mesh elements 'v', 'e' or 'f' on which the layer will operate.
        """
        super(MeshMultiHeadAttention, self).__init__()

        assert d_v % h == 0 and d_e % h == 0 and d_f % h == 0

        # We assume d_v and d_e always equals d_k
        self.d_v_k = d_v_k = d_v // h
        self.d_e_k = d_e_k = d_e // h
        self.d_f_k = d_f_k = d_f // h

        # Number of heads
        self.h = h

        # Add learning mode and symmetry
        self.modes = modes
        self.attn_sym = attn_sym
        self.attn_acc = attn_acc
        self.attn_mask = attn_mask
        self.norm_type = norm_type

        # Initialize vertex, edge, face W, K, V linear layers
        self.v_linears, self.e_linears, self.f_linears = self._initialize_linear_layers(
            modes, attn_sym, d_v, d_e, d_f
        )

        (norm_v_Q, norm_v_K), (norm_e_Q, norm_e_K), (norm_f_Q, norm_f_K) = (
            self._initialize_norm_layers(
                modes, attn_sym, d_v_k, d_e_k, d_f_k, norm_type=norm_type
            )
        )

        self.norm_v_Qs = clones(norm_v_Q, self.h)
        self.norm_v_Ks = clones(norm_v_K, self.h)
        self.norm_e_Qs = clones(norm_e_Q, self.h)
        self.norm_e_Ks = clones(norm_e_K, self.h)
        self.norm_f_Qs = clones(norm_f_Q, self.h)
        self.norm_f_Ks = clones(norm_f_K, self.h)

        # Initialize operator accelerator layers
        self.v_acc = clones(
            AccelerateOperator(d_v_k) if "v" in modes and attn_acc else nn.Identity(),
            self.h,
        )
        self.e_acc = clones(
            AccelerateOperator(d_e_k) if "e" in modes and attn_acc else nn.Identity(),
            self.h,
        )
        self.f_acc = clones(
            AccelerateOperator(d_f_k) if "f" in modes and attn_acc else nn.Identity(),
            self.h,
        )

        # Initialize output linear layers
        self.v_linear_out = nn.Linear(d_v, d_v) if "v" in modes else nn.Identity()
        self.e_linear_out = nn.Linear(d_e, d_e) if "e" in modes else nn.Identity()
        self.f_linear_out = nn.Linear(d_f, d_f) if "f" in modes else nn.Identity()

        self.v_attn = None
        self.e_attn = None
        self.f_attn = None

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_attn = dropout_attn

    def _initialize_linear_layers(self, modes, sym, d_v, d_e, d_f):
        """
        Initialize linear layers based on `modes` and `sym` arguments.
        """
        linears = self.LINEARS_SYM if sym is True else self.LINEARS

        layers = linears[modes[0]]

        for mode in modes[1:]:

            _layers = linears[mode]

            layers = [
                [i or j for i, j in zip(m_i, m_j)] for m_i, m_j in zip(layers, _layers)
            ]

        layer_cls = {1: nn.Linear, 0: nn.Identity}

        kw = {"bias": False}

        v_linears = nn.ModuleList([layer_cls[l](d_v, d_v, **kw) for l in layers[0]])
        e_linears = nn.ModuleList([layer_cls[l](d_e, d_e, **kw) for l in layers[1]])
        f_linears = nn.ModuleList([layer_cls[l](d_f, d_f, **kw) for l in layers[2]])

        return v_linears, e_linears, f_linears

    def _initialize_norm_layers(self, modes, attn_sym, d_v, d_e, d_f, norm_type):
        """
        Initialize norm layers based on `modes` and `attn_sym` arguments.
        """
        norms = self.NORMS_SYM if attn_sym is True else self.NORMS

        layers = norms[modes[0]]

        for mode in modes[1:]:

            _layers = norms[mode]

            layers = [
                [i or j for i, j in zip(m_i, m_j)] for m_i, m_j in zip(layers, _layers)
            ]

        if norm_type is False:
            layers = [[0 for l in space] for space in layers]

        elif norm_type == "rms_norm":
            layers = [[2 if l == 1 else 0 for l in space] for space in layers]

        layer_cls = {0: nn.Identity, 1: nn.LayerNorm, 2: nn.RMSNorm}

        v_norms = [layer_cls[l](d_v) for l in layers[0]]
        e_norms = [layer_cls[l](d_e) for l in layers[1]]
        f_norms = [layer_cls[l](d_f) for l in layers[2]]

        return v_norms, e_norms, f_norms

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

    def forward(self, x_v, x_e, x_f, d_0, d_1, mask=None):
        """
        Forward step of multi head attention.

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

        Returns
        -------
        x_v : torch.Tensor
            Output tensor of shape (n, d_v).
        x_e : torch.Tensor
            Output tensor of shape (m, d_e).
        x_f : torch.Tensor
            Output tensor of shape (k, d_f).
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        nbatches = x_v.size(0)

        # 1. Calculate linear projections in batch
        v_Q, v_K, v_V = [
            linear(x_v).view(nbatches, -1, self.h, self.d_v_k).transpose(1, 2)
            for linear in self.v_linears
        ]

        e_Q, e_K, e_V = [
            linear(x_e).view(nbatches, -1, self.h, self.d_e_k).transpose(1, 2)
            for linear in self.e_linears
        ]

        f_Q, f_K, f_V = [
            linear(x_f).view(nbatches, -1, self.h, self.d_f_k).transpose(1, 2)
            for linear in self.f_linears
        ]

        # Apply normalization across heads
        if self.norm_type:
            v_Q = self._apply_norm_across_heads(self.norm_v_Qs, v_Q)
            e_Q = self._apply_norm_across_heads(self.norm_e_Qs, e_Q)
            f_Q = self._apply_norm_across_heads(self.norm_f_Qs, f_Q)
            v_K = self._apply_norm_across_heads(self.norm_v_Ks, v_K)
            e_K = self._apply_norm_across_heads(self.norm_e_Ks, e_K)
            f_K = self._apply_norm_across_heads(self.norm_f_Ks, f_K)

        if self.attn_sym is True:
            v_K = v_Q
            e_K = e_Q
            f_K = f_Q

        # Apply attention on projected vectors in batch & concat
        # Apply mesh attention on vertices
        if "v" in self.modes:

            mask = d_0 if self.attn_mask is True else None

            x_v, self.v_attn = mesh_attention_v__v2(
                v_Q, v_V, e_K, d_0, dropout=self.dropout_attn, mask=mask
            )

            x_v = (
                x_v.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_v_k)
            )

        # Apply mesh attention on edges
        if "e" in self.modes:
            x_e, self.e_attn = mesh_attention_e__v2(
                v_K, e_Q, e_K, e_V, f_K, d_0, d_1, dropout=self.dropout_attn, mask=None
            )

            x_e = (
                x_e.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_e_k)
            )

        # Apply mesh attention on faces
        if "f" in self.modes:
            x_f, self.f_attn = mesh_attention_f__v2(
                e_Q, f_Q, f_V, d_1, dropout=self.dropout_attn, mask=None
            )

            x_f = (
                x_f.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_f_k)
            )

        del v_Q
        del v_K
        del v_V
        del e_Q
        del e_K
        del e_V
        del f_Q
        del f_K
        del f_V

        return self.v_linear_out(x_v), self.e_linear_out(x_e), self.f_linear_out(x_f)


def mesh_attention_v__v2(v_Q, v_V, e_K, d_0, dropout=None, mask=None, symmetric=False):
    """
    Calculate mesh attention over `n` vertices.

    Parameters
    ----------
    v_Q : torch.Tensor
        Vertex features mapped by Q matrix.
    v_V : torch.Tensor
        Vertex features mapped by V matrix.
    e_K : torch.Tensor
        Edge features mapped by K matrix.
    d_0 : torch.Tensor
        Incidence matrix (m, n) as a boundary operator taking vertices to edges.
    mask : torch.Tensor
        Mask to apply on the attention matrix.
    dropout : None | nn.Module
        A dropout module.

    Returns
    -------
    _x : torch.Tensor
        Result of applying mesh attention over vertices.
    None
        No explicit attention matrix is computed.
    """
    # Get number of heads
    h = v_V.size(1)

    if symmetric is False:
        _x = torch.stack([torch.bmm(d_0, v_V[:, i, ...]) for i in range(h)], dim=1)

        _x, _ = cross_attention(
            _x, v_Q, e_K, dropout=dropout, mask=mask, symmetric=symmetric
        )

    else:
        _x, _ = cross_attention(
            v_V, v_Q, e_K, dropout=dropout, mask=mask, symmetric=symmetric
        )

    return _x, None


def mesh_attention_e__v2(v_K, e_Q, e_K, e_V, f_K, d_0, d_1, dropout=None, mask=None):
    """
    Calculate mesh attention over `m` edges.

    Returns
    -------
    torch.Tensor
        Result of applying mesh attention over edges.
    None
        No explicit attention matrix is computed.
    """
    # Get number of heads
    h = e_V.size(1)

    # e & f
    _x = torch.stack([torch.bmm(d_1, e_V[:, i, ...]) for i in range(h)], dim=1)

    _x, _ = cross_attention(_x, e_Q, f_K, dropout)

    # v & e
    _y, _ = cross_attention(e_V, v_K, e_K, dropout)

    _y = torch.stack([torch.bmm(d_0, _y[:, i, ...]) for i in range(h)], dim=1)

    return _y + _x, None


def mesh_attention_f__v2(e_Q, f_Q, f_V, d_1, dropout=None, mask=None):
    """
    Calculate mesh attention over `k` faces.

    Returns
    -------
    _x : torch.Tensor
        Result of applying mesh attention over vertices.
    None
        No explicit attention matrix is computed.
    """
    # Get number of heads, edges (m) and vertices (n)
    h = f_V.size(1)

    _y, _ = cross_attention(f_V, e_Q, f_Q, dropout)

    _y = torch.stack([torch.bmm(d_1, _y[:, i, ...]) for i in range(h)], dim=1)

    return _y, None


def cross_attention_linear(x, q, k, dropout=None, mask=None, symmetric=False):
    """
    Perform linear cross attention between Q and K.
    """
    pass


def cross_attention(x, q, k, dropout=None, mask=None, symmetric=False):
    """
    Perform cross attention between Q and K.
    """
    bs, h, d_q = q.size(0), q.size(1), q.size(-1)

    if mask is None:
        QK = torch.einsum("bhnd,bhmd->bhnm", q, k)

    else:
        # mask = _coo_to_csr(mask.transpose(-2, -1))
        #
        # QK = torch.stack(
        #     [
        #         torch.sparse.sampled_addmm(
        #             mask, q[:, i, ...], k[:, i, ...].transpose(-1, -2), beta=0.0
        #         ).to_dense()
        #         for i in range(h)
        #     ],
        #     dim=1,
        # )
        mask = abs(mask.transpose(-2, -1)).to_dense()
        QK = torch.stack(
            [
                torch.bmm(q[:, i, ...], k[:, i, ...].transpose(-1, -2)) * mask
                for i in range(h)
            ],
            dim=1,
        )

    QK = QK / math.sqrt(d_q)

    if symmetric is True:
        QK = torch.einsum("bhnm,bhne->bhnn", QK, QK)

    QK = torch.nn.functional.softmax(QK, dim=-1)

    if dropout is not None:
        QK = F.dropout(QK, dropout)

    return torch.einsum("bhnm,bhmd->bhnd", QK, x), QK


def cross_attention_v2(x, q, k, acc, dropout=None):
    """ """
    m = k.size(-2)

    # K.T (d, m) @ x m, d)
    _x = torch.einsum("bhme,bhmd->bhed", k, x) / m

    # Apply accelerator across heads
    _x = torch.stack([acc(_x[:, i, ...]) for i, acc in enumerate(acc)], dim=1)

    # Q (n, d) @ x (d, d)
    _x = torch.einsum("bhne,bhed->bhnd", q, _x)

    return _x, None
