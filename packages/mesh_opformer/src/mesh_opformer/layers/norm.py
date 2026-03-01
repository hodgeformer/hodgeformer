import torch

from torch import nn


class LayerNorm(nn.Module):
    """
    Construct a layernorm module.
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class L1Norm(nn.Module):
    """
    Construct a L1Norm module.
    """

    def __init__(self, eps=1e-6):
        super(L1Norm, self).__init__()
        self.eps = eps

    def forward(self, x):
        norm = torch.linalg.norm(x, dim=-1, ord=1, keepdim=True)

        return x / (norm + self.eps)


class GraphNorm(nn.Module):
    """
    A class to represent a Graph Normalization module.
    """

    def __init__(self, d_k, affine=True, is_node=True, eps=1e-5):
        """
        Initializer for `GraphNorm` instance.

        Parameters
        ----------
        d_k : int
            Number of features.
        affine : bool
            Apply affine transformation.
        is_node : bool
            Whether to
        eps : float
            Epsilon value.
        """
        super(GraphNorm, self).__init__()
        self.d_k = d_k
        self.affine = affine
        self.is_node = is_node
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.d_k))
            self.beta = nn.Parameter(torch.zeros(self.d_k))

        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (var + self.eps)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method.
        """
        norm_x = self.norm(x)

        if self.affine:
            return self.gamma * norm_x + self.beta
        else:
            return norm_x
