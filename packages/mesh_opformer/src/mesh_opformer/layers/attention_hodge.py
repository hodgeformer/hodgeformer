from typing import List, Tuple

import math
import torch
import torch.nn as nn

from einops import repeat

from .utils import AccelerateOperator
from .utils import clones

from .norm import L1Norm

from .parametrizations import degree_matrix

from .attention import (
    linear_attention__flowformer,
    linear_attention__fast_transformers,
    linear_attention__efficient,
)

from .attention import sparse_attention

# Some notes on lumping operations
# Let x be a vector of 0-forms
# np.eye(x.size) * ((np.ones_like(x) @ np.ones_like(x).T) @ A)


class MeshMultiHeadHodgeAttention(nn.Module):
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
        hodge_type="grouped",
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
        hodge_type : {'diag', 'full', 'full-linear', 'grouped'}
            Whether to calculate a diagonal, a full or a full linear hodge matrix.
        attn_acc : bool
            Whether to use accelerator operator.
        modes : Tuple[str]
            Mesh elements 'v', 'e' or 'f' on which the layer will operate.
        """
        super(MeshMultiHeadHodgeAttention, self).__init__()

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
        self.hodge_type = hodge_type

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
        if norm_type and norm_type not in ("layer_norm", "rms_norm", "l1_norm"):
            raise ValueError(
                (
                    "Given `norm_type` argument is not valid. "
                    "Choose from ('layer_norm', 'rms_norm', 'l1_norm')."
                )
            )
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

        elif norm_type == "l1_norm":
            layers = [[3 if l == 1 else 0 for l in space] for space in layers]

        layer_cls = {0: nn.Identity, 1: nn.LayerNorm, 2: nn.RMSNorm, 3: L1Norm}
        layer_kws = {
            1: {"elementwise_affine": False, "bias": False},
        }

        v_norms = [layer_cls[l](d_v, **layer_kws.get(l, {})) for l in layers[0]]
        e_norms = [layer_cls[l](d_e, **layer_kws.get(l, {})) for l in layers[1]]
        f_norms = [layer_cls[l](d_f, **layer_kws.get(l, {})) for l in layers[2]]

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

    def forward(
        self, x_v, x_e, x_f, d_0, d_1, v_idx=None, e_idx=None, f_idx=None, mask=None
    ):
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
        v_idx : torch.Tensor | None
            Indices indicating a local vertex neighborhood as tensor of shape (n, k_v).
        e_idx : torch.Tensor | None
            Indices indicating a local edge neighborhood as tensor of shape (m, k_e).
        f_idx : torch.Tensor | None
            Indices indicating a local face neighborhood as tensor of shape (k, k_f).
        mask : torch.Tensor | None
            A mask tensor. Not supported for now.

        Returns
        -------
        x_v : torch.Tensor
            Output tensor of shape (n, d_v).
        x_e : torch.Tensor
            Output tensor of shape (m, d_e).
        x_f : torch.Tensor
            Output tensor of shape (k, d_f).
        """
        # NOTE 1: Cao applies has a hyperparameter for applying a normalization
        # layer HERE, either `LayerNorm` or `InstanceNorm`.

        # NOTE 2: In `lucidrains` linear transformer implementation, after the
        # Key value multiplication we divide by the sequence length. Here the
        # corresponding variable is number of nodes, edges and faces.

        # NOTE 3.
        # Rethink about attention. We use attention between edges and between vertices
        # to create learnable hodge matrices. Apply sparsity restrictions to ensure locality
        #
        # NOTE 4.
        # Test initialization Q, V with as orthogonal weight matrices

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        nbatches = x_v.size(0)

        # Calculate `q, k, v` linear projections
        v_Q, v_K, v_V = [
            linear(x_v).view(nbatches, -1, self.h, self.d_v_k).transpose(1, 2)
            for linear in self.v_linears
        ]

        if self.norm_type:
            v_Q = self._apply_norm_across_heads(self.norm_v_Qs, v_Q)
            v_K = self._apply_norm_across_heads(self.norm_v_Ks, v_K)

        e_Q, e_K, e_V = [
            linear(x_e).view(nbatches, -1, self.h, self.d_e_k).transpose(1, 2)
            for linear in self.e_linears
        ]

        if self.norm_type:
            e_Q = self._apply_norm_across_heads(self.norm_e_Qs, e_Q)
            e_K = self._apply_norm_across_heads(self.norm_e_Ks, e_K)

        f_Q, f_K, f_V = [
            linear(x_f).view(nbatches, -1, self.h, self.d_f_k).transpose(1, 2)
            for linear in self.f_linears
        ]

        if self.norm_type:
            f_Q = self._apply_norm_across_heads(self.norm_f_Qs, f_Q)
            f_K = self._apply_norm_across_heads(self.norm_f_Ks, f_K)

        if self.attn_sym is True:
            v_K = v_Q
            e_K = e_Q
            f_K = f_Q

        # Apply attention on projected vectors in batch & concat
        # Apply mesh attention on vertices
        if "v" in self.modes:
            x_v, self.v_attn = mesh_attention_v(
                v_Q,
                v_K,
                v_V,
                e_Q,
                e_K,
                d_0,
                v_idx,
                e_idx,
                self.v_acc,
                hodge_type=self.hodge_type,
                dropout=self.dropout_attn,
            )

            x_v = (
                x_v.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_v_k)
            )

        # Apply mesh attention on edges
        if "e" in self.modes:
            x_e, self.e_attn = mesh_attention_e(
                v_Q,
                v_K,
                e_Q,
                e_K,
                e_V,
                f_Q,
                f_K,
                d_0,
                d_1,
                v_idx,
                e_idx,
                f_idx,
                self.e_acc,
                hodge_type=self.hodge_type,
                dropout=self.dropout_attn,
            )

            x_e = (
                x_e.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_e_k)
            )

        # Apply mesh attention on faces
        if "f" in self.modes:
            x_f, self.f_attn = mesh_attention_f(
                e_Q,
                e_K,
                f_Q,
                f_K,
                f_V,
                d_1,
                e_idx,
                f_idx,
                self.f_acc,
                hodge_type=self.hodge_type,
                dropout=self.dropout_attn,
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


def mesh_attention_v(
    v_Q: torch.Tensor,
    v_K: torch.Tensor,
    v_V: torch.Tensor,
    e_Q: torch.Tensor,
    e_K: torch.Tensor,
    d_0: torch.Tensor,
    v_idx: torch.Tensor,
    e_idx: torch.Tensor,
    v_acc: torch.Tensor,
    hodge_type: str = "diag",
    dropout: float | None = None,
) -> Tuple[torch.Tensor, None]:
    """
    Calculate mesh attention over `n` vertices.

                        *0^(-1) @ d_0.T @ *1 @ d_0

    Parameters
    ----------
    v_Q : torch.Tensor
        Vertex features mapped by Q matrix.
    v_K : torch.Tensor
        Vertex features mapped by K matrix.
    v_V : torch.Tensor
        Vertex features mapped by V matrix.
    e_Q : torch.Tensor
        Edge features mapped by Q matrix.
    e_K : torch.Tensor
        Edge features mapped by K matrix.
    d_0 : torch.Tensor
        Incidence matrix (m, n) as a boundary operator taking vertices to edges.
    v_idx : torch.Tensor
        Indices indicating a local vertex neighborhood as tensor of shape (n, k_v).
    e_idx : torch.Tensor
        Indices indicating a local edge neighborhood as tensor of shape (m, k_e).
    hodge_type : {'diag', 'full', 'full-linear', 'grouped'}
        Whether to calculate a diagonal, a full, linear or grouped hodge matrix.
    dropout : None | nn.Module
        A dropout module.

    Returns
    -------
    _x : torch.Tensor
        Result of applying mesh attention over vertices.
    None
        No explicit attention matrix is computed.
    """
    # Get number of heads, edges (m) and vertices (n)
    h = v_Q.size(1)

    _x = v_V

    with torch.autocast(v_V.device.type, dtype=torch.bfloat16, enabled=False):
        _x = _x.float()
        _x = torch.stack([torch.bmm(d_0, _x[:, i, ...]) for i in range(h)], dim=1)

    if hodge_type == "diag":
        _x = hodge_diagonal_ops(_x, e_Q, e_K, inv=False, dropout=dropout)
    elif hodge_type == "full":
        _x = hodge_full_ops(_x, e_Q, e_K, inv=False, dropout=dropout)
    elif hodge_type == "full-linear":
        _x = hodge_full_ops_linear(_x, e_Q, e_K, inv=False, dropout=dropout)
    elif hodge_type == "grouped":
        _x = hodge_grouped(_x, e_Q, e_K, e_idx, inv=False, dropout=dropout)

    with torch.autocast(_x.device.type, dtype=torch.bfloat16, enabled=False):
        _x = _x.float()
        _x = torch.stack(
            [torch.bmm(d_0.transpose(-2, -1), _x[:, i, ...]) for i in range(h)], dim=1
        )

    if hodge_type == "diag":
        _x = hodge_diagonal_ops(_x, v_Q, v_K, inv=True, dropout=dropout)
    elif hodge_type == "full":
        _x = hodge_full_ops(_x, v_Q, v_K, inv=True, dropout=dropout)
    elif hodge_type == "full-linear":
        _x = hodge_full_ops_linear(_x, v_Q, v_K, inv=True, dropout=dropout)
    elif hodge_type == "grouped":
        _x = hodge_grouped(_x, v_Q, v_K, v_idx, inv=True, dropout=dropout)

    return _x, None


def mesh_attention_e(
    v_Q: torch.Tensor,
    v_K: torch.Tensor,
    e_Q: torch.Tensor,
    e_K: torch.Tensor,
    e_V: torch.Tensor,
    f_Q: torch.Tensor,
    f_K: torch.Tensor,
    d_0: torch.Tensor,
    d_1: torch.Tensor,
    v_idx: torch.Tensor,
    e_idx: torch.Tensor,
    f_idx: torch.Tensor,
    e_acc: List[torch.nn.Module],
    hodge_type: str = "diag",
    dropout: float | None = None,
) -> Tuple[torch.Tensor, None]:
    """
    Calculate mesh attention over `m` edges.

            d_0 @ *0^(-1) @ d_0.T @ *1  +  *1^(-1) @ d_1.T @ *2 @ d_1

    Parameters
    ----------
    v_Q : torch.Tensor
        Vertex features mapped by Q matrix.
    v_K : torch.Tensor
        Vertex features mapped by K matrix.
    e_Q : torch.Tensor
        Edge features mapped by Q matrix.
    e_K : torch.Tensor
        Edge features mapped by K matrix.
    e_V : torch.Tensor
        Edge features mapped by V matrix.
    f_Q : torch.Tensor
        Face features mapped by Q matrix.
    f_K : torch.Tensor
        Face features mapped by K matrix.
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
    hodge_type : {'diag', 'full', 'full-linear', 'grouped'}
        Whether to calculate a diagonal, a full, linear or grouped hodge matrix.
    dropout : None | nn.Module
        A dropout module.

    Returns
    -------
    _x : torch.Tensor
        Result of mesh attention
    None
        No explicit attention matrix is computed.
    """
    # Get number of heads (h), edges (m), vertices (n), faces (k)
    h = e_V.size(1)
    m = d_0.size(1)
    n = d_0.size(2)
    k = d_1.size(1)

    _x = _y = e_V

    # A. Calculate `*1^(-1) @ d_1.T @ *2 @ d_1`
    with torch.autocast(e_V.device.type, dtype=torch.float32, enabled=False):
        _x = _x.float()
        _x = torch.stack([torch.bmm(d_1, _x[:, i, ...]) for i in range(h)], dim=1)

    if hodge_type == "diag":
        _x = hodge_diagonal_ops(_x, f_Q, f_K, inv=False, dropout=dropout)
    elif hodge_type == "full":
        _x = hodge_full_ops(_x, f_Q, f_K, inv=False, dropout=dropout)
    elif hodge_type == "full-linear":
        _x = hodge_full_ops_linear(_x, f_Q, f_K, inv=False, dropout=dropout)
    elif hodge_type == "grouped":
        _x = hodge_grouped(_x, f_Q, f_K, f_idx, inv=False, dropout=dropout)

    with torch.autocast(_x.device.type, dtype=torch.float32, enabled=False):
        _x = _x.float()
        _x = torch.stack(
            [torch.bmm(d_1.transpose(-2, -1), _x[:, i, ...]) for i in range(h)], dim=1
        )

    if hodge_type == "diag":
        _x = hodge_diagonal_ops(_x, e_Q, e_K, inv=True, dropout=dropout)
    elif hodge_type == "full":
        _x = hodge_full_ops(_x, e_Q, e_K, inv=True, dropout=dropout)
    elif hodge_type == "full-linear":
        _x = hodge_full_ops_linear(_x, e_Q, e_K, inv=True, dropout=dropout)
    elif hodge_type == "grouped":
        _x = hodge_grouped(_x, e_Q, e_K, e_idx, inv=True, dropout=dropout)

    # B. Calculate `d_0 @ *0^(-1) @ d_0.T @ *1`
    if hodge_type == "diag":
        _y = hodge_diagonal_ops(_y, e_Q, e_K, inv=False, dropout=dropout)
    elif hodge_type == "full":
        _y = hodge_full_ops(_y, e_Q, e_K, inv=False, dropout=dropout)
    elif hodge_type == "full-linear":
        _y = hodge_full_ops_linear(_y, e_Q, e_K, inv=False, dropout=dropout)
    elif hodge_type == "grouped":
        _y = hodge_grouped(_y, e_Q, e_K, e_idx, inv=False, dropout=dropout)

    with torch.autocast(_y.device.type, dtype=torch.float32, enabled=False):
        _y = _y.float()
        _y = torch.stack(
            [torch.bmm(d_0.transpose(-2, -1), _y[:, i, ...]) for i in range(h)], dim=1
        )

    if hodge_type == "diag":
        _y = hodge_diagonal_ops(_y, v_Q, v_K, inv=True, dropout=dropout)
    elif hodge_type == "full":
        _y = hodge_full_ops(_y, v_Q, v_K, inv=True, dropout=dropout)
    elif hodge_type == "full-linear":
        _y = hodge_full_ops_linear(_y, v_Q, v_K, inv=True, dropout=dropout)
    elif hodge_type == "grouped":
        _y = hodge_grouped(_y, v_Q, v_K, v_idx, inv=True, dropout=dropout)

    with torch.autocast(_y.device.type, dtype=torch.float32, enabled=False):
        _y = _y.float()
        _y = torch.stack([torch.bmm(d_0, _y[:, i, ...]) for i in range(h)], dim=1)

    return _y + _x, None


def mesh_attention_f(
    e_Q: torch.Tensor,
    e_K: torch.Tensor,
    f_Q: torch.Tensor,
    f_K: torch.Tensor,
    f_V: torch.Tensor,
    d_1: torch.Tensor,
    e_idx: torch.Tensor,
    f_idx: torch.Tensor,
    f_acc: List[torch.nn.Module],
    hodge_type: str = "diag",
    dropout: float | None = None,
) -> Tuple[torch.Tensor, None]:
    """
    Calculate mesh attention over `m` edges.

                            d_1 @ *1^(-1) @ d_1.T @ *2

    Parameters
    ----------
    e_Q : torch.Tensor
        Edge features mapped by Q matrix.
    e_K : torch.Tensor
        Edge features mapped by K matrix.
    f_Q : torch.Tensor
        Face features mapped by Q matrix.
    f_K : torch.Tensor
        Face features mapped by K matrix.
    f_V : torch.Tensor
        Face features mapped by V matrix.
    d_1 : torch.Tensor
        Incidence matrix (k, m) as a boundary operator taking edges to faces.
    e_idx : torch.Tensor
        Indices indicating a local edge neighborhood as tensor of shape (m, k_e).
    f_idx : torch.Tensor
        Indices indicating a local face neighborhood as tensor of shape (k, k_f).
    hodge_type : {'diag', 'full', 'full-linear', 'grouped'}
        Whether to calculate a diagonal, a full, linear or grouped hodge matrix.
    dropout : None | nn.Module
        A dropout module.

    Returns
    -------
    _x : torch.Tensor
        Result of mesh attention
    None
        No explicit attention matrix is computed.
    """
    # Get number of heads (h),
    h = f_V.size(1)

    _y = f_V

    # Calculate `d_1 @ *1^(-1) @ d_1.T @ *2`
    if hodge_type == "diag":
        _y = hodge_diagonal_ops(_y, f_Q, f_K, inv=False, dropout=dropout)
    elif hodge_type == "full":
        _y = hodge_full_ops(_y, f_Q, f_K, inv=False, dropout=dropout)
    elif hodge_type == "full-linear":
        _y = hodge_full_ops_linear(_y, f_Q, f_K, inv=False, dropout=dropout)
    elif hodge_type == "grouped":
        _y = hodge_grouped(_y, f_Q, f_K, f_idx, inv=False, dropout=dropout)

    with torch.autocast(_y.device.type, dtype=torch.float32, enabled=False):
        _y = _y.float()
        _y = torch.stack(
            [torch.bmm(d_1.transpose(-2, -1), _y[:, i, ...]) for i in range(h)], dim=1
        )

    if hodge_type == "diag":
        _y = hodge_diagonal_ops(_y, e_Q, e_K, inv=True, dropout=dropout)
    elif hodge_type == "full":
        _y = hodge_full_ops(_y, e_Q, e_K, inv=True, dropout=dropout)
    elif hodge_type == "full-linear":
        _y = hodge_full_ops_linear(_y, e_Q, e_K, inv=True, dropout=dropout)
    elif hodge_type == "grouped":
        _y = hodge_grouped(_y, e_Q, e_K, e_idx, inv=True, dropout=dropout)

    with torch.autocast(_y.device.type, dtype=torch.float32, enabled=False):
        _y = _y.float()
        _y = torch.stack([torch.bmm(d_1, _y[:, i, ...]) for i in range(h)], dim=1)

    return _y, None


def hodge_diagonal_ops(x, q, k, inv=True, eps=1e-6, dropout=None):
    """
    Build diagonal discrete Hodge operator and apply to input `x`.
    """
    d_q = q.size(-1)

    h = torch.einsum("bhmd,bhmd->bhm", q, k) / math.sqrt(d_q)

    if inv is True:
        h = torch.reciprocal(h + eps)

    if dropout is not None:
        h = nn.functional.dropout(h, p=dropout)

    return torch.mul(h.unsqueeze(-1), x)


def hodge_full_ops(x, q, k, inv=True, dropout=None):
    """
    Apply discrete Hodge operator to input `x`.
    """
    _, h, _, d = q.size()

    if inv is False:
        qk = torch.einsum("bhnd,bhmd->bhnm", q, k) / math.sqrt(d)

    elif inv is True:
        qk = torch.einsum("bhnd,bhmd->bhmn", q, k) / math.sqrt(d)

    qk = torch.nn.functional.softmax(qk, dim=-1)

    if dropout is not None:
        qk = nn.functional.dropout(qk, p=dropout)

    return torch.einsum("bhlk,bhkd->bhld", qk, x)


def hodge_full_ops_linear(x, q, k, inv=True, dropout=None):
    """
    Apply discrete Hodge operator to input `x` from right to left.

    This is linear attention proposed by Katharopoulos et al., 2020 in
    `Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention`
    """
    if inv is False:
        _q, _k = q, k
    else:
        _q, _k = k, q

    x = linear_attention__flowformer(x, _q, _k)

    if dropout is not None:
        x = nn.functional.dropout(x, p=dropout)

    return x


def hodge_grouped(x, q, k, idx, inv=False, dropout=None):
    """
    Apply sparse attention.
    """
    if inv is False:
        _q, _k = q, k
    else:
        _q, _k = k, q

    return sparse_attention(x, _q, _k, idx=idx, dropout=dropout)


def cross_attention(x, q, k):
    """
    Perform cross attention between Q and K.
    """
    d_q = q.size(-1)

    qk = torch.einsum("bhnd,bhmd->bhnm", q, k) / math.sqrt(d_q)
    qk = torch.nn.functional.softmax(qk, dim=-1)

    return torch.einsum("bhnm,bhmd->bhnd", qk, x), qk
