from typing import Tuple, Dict, List

import torch
from torch import nn

from .fc import MLPTwoLayer, MLPSimple
from .pooling import (
    MultiHeadAttentionGatedPooling,
    MultiHeadAttentionWeightedAverage,
    AdditiveAttentionWeightedAverage,
)


def initialize_dot_attention_pooling(d: int, modes: str, **kw) -> nn.ModuleDict:
    """
    Initialze dot attention pooling layers.
    """
    _kw = {"h": 4, "d": d, "dropout_attn": 0.1}
    _kw.update(kw)

    return nn.ModuleDict(
        {mode: MultiHeadAttentionWeightedAverage(**_kw) for mode in modes}
    )


def initialize_add_attention_pooling(d: int, modes: str, **kw) -> nn.ModuleDict:
    """
    Initialze additive attention pooling layers.
    """
    _kw = {"h": 4, "d": d, "dropout_attn": 0.1}
    _kw.update(kw)

    return nn.ModuleDict(
        {mode: AdditiveAttentionWeightedAverage(**_kw) for mode in modes}
    )


def initialize_mil_attention_pooling(d: int, modes: str, **kw) -> nn.ModuleDict:
    """
    Initialize MIL-like attention pooling layer.
    """
    _kw = {"h": 4, "d": d, "dropout_attn": 0.1}
    _kw.update(kw)

    return nn.ModuleDict(
        {mode: MultiHeadAttentionGatedPooling(**_kw) for mode in modes}
    )


POOLING_HEADS = {
    "attention-dot": initialize_dot_attention_pooling,
    "attention-add": initialize_add_attention_pooling,
    "attention-mil": initialize_mil_attention_pooling,
}


class MeshClassifier(nn.Module):
    """
    A class representing a classifier head to be placed on top of
    an encoder.
    """

    def __init__(
        self,
        encoder,
        d_emb=128,
        num_classes=2,
        dropout=0.1,
        modes="ve",
        head_type="linear",
        head_kw: Dict = {},
    ):
        """
        Initializer for `MeshClassifier`.

        Parameters
        ----------
        encoder : nn.Module
            An encoder model encoding vertices, edges and/or faces.
        num_classes : int
            Number of output classes.
        d_emb : int
            The embedding dimension. For now is considered the same
            for nodes, edges and faces.
        modes : str
            The mesh elements 'v', 'e' or 'f' on which output classes
            will be calculated.
        """
        super(MeshClassifier, self).__init__()

        HEAD_TYPES = ("linear",) + tuple(POOLING_HEADS.keys())

        if head_type not in HEAD_TYPES:
            raise ValueError(
                "`head_type` arg not valid. Choose from {}.".format(HEAD_TYPES)
            )

        self.modes = modes

        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head_type = head_type

        self.l_head = nn.Linear(d_emb * len(modes), num_classes)

        if head_type in POOLING_HEADS:
            self.t_heads = POOLING_HEADS[head_type](d_emb, modes, **head_kw)

    def forward(self, *args, **kwargs):
        """
        Encode input, aggregate features and apply classification head.
        """
        x_v, x_e, x_f = self.encoder(*args, **kwargs)

        X_OUT = {"v": x_v, "e": x_e, "f": x_f}

        if self.head_type == "linear":
            x_out = torch.concat(
                [torch.mean(X_OUT[mode], dim=1) for mode in self.modes], dim=1
            )

        elif self.head_type in POOLING_HEADS:
            x_out = torch.concat(
                [self.t_heads[mode](X_OUT[mode]) for mode in self.modes], dim=1
            )

        x_out = self.dropout(x_out)
        logits = self.l_head(x_out)

        return logits


class MeshSegmenter(nn.Module):
    """
    A class representing a segmentation head to be placed on top of
    an encoder.
    """

    def __init__(
        self, encoder, num_classes=2, d_emb=128, dropout=0.1, out="f", modes="v"
    ):
        """
        Initializer for `MeshSegmenter`.

        Parameters
        ----------
        encoder : nn.Module
            An encoder model encoding vertices, edges and/or faces.
        num_classes : int
            Number of output classes per vertex.
        d_emb : int
            The embedding dimension. For now is considered the same
            for nodes, edges and faces.
        dropout : float
            Dropout rate before linear map.
        out : str
            Mesh element on which the segmentation label is applied.
        modes : str
            The mesh elements 'v', 'e' or 'f' on which output classes
            will be calculated.
        """
        super(MeshSegmenter, self).__init__()

        self.out = out
        self.modes = modes

        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(d_emb * len(modes), num_classes)

    def forward(self, *args, **kwargs):
        """
        Encode input, aggregate features and apply classification head.
        """
        x_v, x_e, x_f = self.encoder(*args, **kwargs)

        d_0, d_1 = args[3], args[4]

        X_OUT = {"v": x_v, "e": x_e, "f": x_f}
        F_OUT = {"v": self.v_to_out, "e": self.e_to_out, "f": self.f_to_out}

        x_out = torch.concat(
            [F_OUT[mode](X_OUT[mode], d_0, d_1, self.out) for mode in self.modes],
            dim=-1,
        )

        x_out = self.dropout(x_out)

        logits = self.linear_out(x_out)

        return logits.transpose(-2, -1)

    def v_to_out(self, x_v, d_0, d_1, out):
        """
        Map vertex features to other mesh elements.
        """
        with torch.autocast(x_v.device.type, dtype=torch.bfloat16, enabled=False):
            x_v = x_v.float()

            if out == "v":
                return x_v

            if out == "e":
                return torch.bmm(torch.abs(d_0), x_v) / 2

            if out == "f":
                x_v = torch.bmm(torch.abs(d_0), x_v)
                return torch.bmm(torch.abs(d_1), x_v) / 6

    def e_to_out(self, x_e, d_0, d_1, out):
        """
        Map edge features to other mesh elements.
        """
        with torch.autocast(x_e.device.type, dtype=torch.bfloat16, enabled=False):
            x_e = x_e.float()

            if out == "v":
                raise NotImplementedError

            if out == "e":
                return x_e

            if out == "f":
                return torch.bmm(torch.abs(d_1), x_e) / 3

    def f_to_out(self, x_f, d_0, d_1, out):
        """
        Map edge features to other mesh elements.
        """
        if out == "v":
            raise NotImplementedError

        if out == "e":
            raise NotImplementedError

        if out == "f":
            return x_f


def compute_correspondence(
    M_F: torch.Tensor,
    N_G: torch.Tensor,
    phi_M: torch.Tensor,
    phi_N: torch.Tensor,
    lambda_M: torch.Tensor,
    lambda_N: torch.Tensor,
    lambda_param: float = 1e-3,
) -> torch.Tensor:
    """
    Computes the functional map correspondence matrix C given features from two shapes.
    Has no trainable parameters.

    The calculations and variable naming follows Section 4 from Donati et al. (2020)
    `Deep Geometric Functional Maps: Robust Feature Learning for Shape Correspondence`

    We denote the following:
    - `b`: batch size
    - `n_M`: no. of vertices for shape `M`
    - `n_N`: no. of vertices for shape `N`
    - `d`: feature dimensions
    - `k`: eigenfunction dimensions

    Parameters
    ----------
    M_F : torch.Tensor
        A tensor of shape `(b, n_M, d)` representing raw-data features for shape `M`.
    N_G : torch.Tensor
        A tensor of shape `(b, n_N, d)` representing raw-data features for shape `N`,
    phi_M : torch.Tensor
        A tensor of shape `(b, n_M, k)` representing eigenfunctions for shape `M`.
    phi_N : torch.Tensor
        A tensor of shape `(b, n_N, k)` representing eigenfunctions for shape `N`.
    lambda_M : torch.Tensor
        A tensor of shape `(b, k)` representing eigenvalues for shape `M`.
    lambda_N : torch.Tensor
        A tensor of shape `(b, k)` representing eigenvalues for shape `N`.
    """
    b, k = lambda_M.size()

    M_F_hat = torch.bmm(phi_M.transpose(1, 2), M_F)  # (k, k)
    N_G_hat = torch.bmm(phi_N.transpose(1, 2), N_G)  # (k, k)

    A, B = M_F_hat, N_G_hat

    Delta = (
        torch.repeat_interleave(lambda_M.unsqueeze(1), repeats=k, dim=1)
        - torch.repeat_interleave(lambda_N.unsqueeze(2), repeats=k, dim=2)
    ) ** 2

    A_t = A.transpose(1, 2)
    A_A_t = torch.bmm(A, A_t)
    B_A_t = torch.bmm(B, A_t)

    C_i = []

    for i in range(k):

        D_i = torch.cat(
            [torch.diag(Delta[_b, i, :].flatten()).unsqueeze(0) for _b in range(b)],
            dim=0,
        )
        C = torch.bmm(
            torch.inverse(A_A_t + lambda_param * D_i),
            B_A_t[:, i, :].unsqueeze(1).transpose(1, 2),
        )
        C_i.append(C.transpose(1, 2))

    C = torch.cat(C_i, dim=1)

    return C


class MeshFunctionalMapper(torch.nn.Module):
    """
    A class to wrap the HodgeFormer model to compute the functional map
    matrix representation.
    """

    def __init__(self, encoder, lambda_param: float = 1e-3):
        """
        Initialize the functional correspondence map
        """
        super(MeshFunctionalMapper, self).__init__()

        self.encoder = encoder

        self.lambda_param = lambda_param

    def forward(
        self,
        tensors_s: torch.Tensor,
        tensors_t: torch.Tensor,
        evecs_s: torch.Tensor,
        evecs_t: torch.Tensor,
        evals_s: torch.Tensor,
        evals_t: torch.Tensor,
        mask: None | torch.Tensor = None,
    ) -> Tuple:
        """
        Forward pass of the
        """
        x_v_s, _, _ = self.encoder(*tensors_s, mask)
        x_v_t, _, _ = self.encoder(*tensors_t, mask)

        C_pred = compute_correspondence(
            M_F=x_v_s,
            N_G=x_v_t,
            phi_M=evecs_s,
            phi_N=evecs_t,
            lambda_M=evals_s,
            lambda_N=evals_t,
            lambda_param=self.lambda_param,
        )

        return C_pred, x_v_s, x_v_t
