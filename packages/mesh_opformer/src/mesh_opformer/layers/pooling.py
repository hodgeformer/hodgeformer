from typing import Dict

import torch

from torch import nn


# NOTE: Also need to check conjuctive Pooling:
# https://github.com/JAEarly/MILTimeSeriesClassification/blob/master/millet/model/pooling.py


class MultiHeadAttentionGatedPooling(nn.Module):
    """
    A class to represent an attention-based gated pooling layer,
    similar to Ilse et al., 2018 in 'Attention-based Deep Multiple
    Instance Learning', https://arxiv.org/pdf/1802.04712.
    """

    def __init__(self, h: int, d: int, dropout_attn: float | None = None) -> None:
        """
        Initializer for the `MultiHeadAttentionWeightedAverage` class.
        """
        super(MultiHeadAttentionGatedPooling, self).__init__()

        assert d % h == 0

        self.h = h

        self.d_k = d_k = d // h

        self.query = nn.Parameter(torch.randn(h, 1, d_k))

        self.W_k = nn.Linear(d, d, bias=False)
        self.W_g = nn.Linear(d, d, bias=False)

        # Not used
        self.dropout_attn = dropout_attn

    def forward(self, x):
        """
        Forward step of multi-head attention gated pooling.

        Parameters
        ----------
        x_v : torch.Tensor
            Features as tensor of shape (n, d).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (n, d).
        """
        b = x.size(0)

        # Calculate `q, k, v` linear projections
        K = self.W_k(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
        G = self.W_g(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
        x = x.view(b, -1, self.h, self.d_k).transpose(1, 2)

        Q = self.query.unsqueeze(0).expand(b, -1, -1, -1)

        # Apply attention on projected vectors in batch & concat
        x = self.multi_head_attention_gated_pooling(x=x, q=Q, k=K, g=G)

        x = x.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)

        del K
        del G

        return x.squeeze()

    def multi_head_attention_gated_pooling(self, x, q, k, g):
        """
        Perform gated attention pooling as in Ilse et al. 2018:

                a = q @ (tanh(V @ x.T) * sigmoid(G @ x.T)).T
                z = a @ x

        Parameters
        ----------

        """
        _k = torch.mul(torch.tanh(k), torch.sigmoid(g))

        _a = torch.einsum("bhmd,bhnd->bhmn", q, _k)
        _a = torch.nn.functional.softmax(_a, dim=-1)

        return torch.einsum("bhmn,bhnd->bhmd", _a, x)


class MultiHeadAttentionWeightedAverage(nn.Module):
    """
    A class to represent an attention-based pooling layer.
    """

    def __init__(self, h: int, d: int, dropout_attn: float | None = None) -> None:
        """
        Initializer for the `MultiHeadAttentionWeightedAverage` class.
        """
        super(MultiHeadAttentionWeightedAverage, self).__init__()

        assert d % h == 0

        self.h = h

        self.d_k = d_k = d // h

        # initialize `n_l = 2` linear mappings
        self.linears = self._initialize_linear_layers(d, n_l=2)

        self.dropout_attn = dropout_attn

        self.query = nn.Parameter(torch.randn(h, 1, d_k))
        self.scale = d_k**0.5

        self.linear_out = nn.Linear(d, d)

    def _initialize_linear_layers(self, d: int, n_l: int = 2) -> nn.ModuleList:
        """
        Initialize linear layers based on `modes` and `sym` arguments.
        """
        kw = {"bias": False}

        return nn.ModuleList([nn.Linear(d, d, **kw) for _ in range(n_l)])

    def forward(self, x):
        """
        Forward step of multi head attention.

        Parameters
        ----------
        x_v : torch.Tensor
            Features as tensor of shape (n, d).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (n, d).
        """
        b = x.size(0)

        # Calculate `q, k, v` linear projections
        K, V = [
            linear(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
            for linear in self.linears
        ]

        Q = self.query.unsqueeze(0).expand(b, -1, -1, -1)

        # Apply attention on projected vectors in batch & concat
        x = self.multi_head_dot_attention(q=Q, k=K, v=V, dropout=self.dropout_attn)

        x = x.transpose(1, 2).contiguous().view(b, -1, self.h * self.d_k)

        del K
        del V

        return self.linear_out(x).squeeze()

    def multi_head_dot_attention(self, q, k, v, dropout=None):
        """
        Perform attention pooling
        """
        A_0 = torch.einsum("bhmd,bhnd->bhmn", q, k) / self.scale

        A_0 = torch.nn.functional.softmax(A_0, dim=-1)

        if dropout is not None:
            A_0 = nn.functional.dropout(A_0, p=dropout)

        return torch.einsum("bhmn,bhnd->bhmd", A_0, v)


class AdditiveAttentionWeightedAverage(nn.Module):
    """ """

    def __init__(self, h: int, d: int, dropout_attn: float | None = None):
        """ """
        super(AdditiveAttentionWeightedAverage, self).__init__()

        self.h = h
        self.d = d

        self.dropout = dropout_attn

        self.query = nn.Parameter(torch.randn(1, d))

        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        """
        Forward step of multi head attention.

        Parameters
        ----------
        x_v : torch.Tensor
            Features as tensor of shape (n, d).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (n, d).
        """
        b = x.size(0)

        # Calculate `q, k`` linear projections
        q = self.query.expand(b, -1).unsqueeze(1)  # (b, 1, d)

        proj_q = self.W_q(q)
        proj_k = self.W_k(x)

        scores = self.v(torch.tanh(proj_q + proj_k))  # (b, n, 1)
        a_weights = torch.softmax(scores, dim=1)  # (b, n, 1)

        if self.dropout is not None:
            a_weights = nn.functional.dropout(a_weights, p=self.dropout)

        return torch.einsum("bmn,bmd->bnd", a_weights, x).squeeze()
