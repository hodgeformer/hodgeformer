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


# class AdjacencyNorm(nn.Module):
#     """ """

#     def __init__(self, num_features, affine=True, eps=1e-5):
#         super(AdjacencyNorm, self).__init__()
#         self.eps = eps
#         self.affine = affine
#         self.num_features = num_features

#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(self.num_features))
#             self.beta = nn.Parameter(torch.zeros(self.num_features))
#         else:
#             self.register_parameter("gamma", None)
#             self.register_parameter("beta", None)


# # Adjance norm for node
# class AdjaNodeNorm(nn.Module):

#     def __init__(self, num_features, eps=1e-5, affine=True):
#         super(AdjaNodeNorm, self).__init__()
#         self.eps = eps
#         self.affine = affine
#         self.num_features = num_features

#         if self.affine:
#             self.gamma = nn.Parameter(torch.ones(self.num_features))
#             self.beta = nn.Parameter(torch.zeros(self.num_features))
#         else:
#             self.register_parameter("gamma", None)
#             self.register_parameter("beta", None)

#     def message_func(self, edges):
#         return {"h": edges.src["norm_h"]}

#     def reduce_func(self, nodes):
#         dst_h = nodes.mailbox["h"]
#         src_h = nodes.data["h"]

#         h = torch.cat([dst_h, src_h.unsqueeze(1)], 1)
#         mean = torch.mean(h, dim=(1, 2))
#         var = torch.std(h, dim=(1, 2))

#         mean = mean.unsqueeze(1).expand_as(src_h)
#         var = var.unsqueeze(1).expand_as(src_h)
#         return {"norm_mean": mean, "norm_var": var}

#     def forward(self, g, h):
#         g.ndata["norm_h"] = h
#         g.update_all(self.message_func, self.reduce_func)

#         mean = g.ndata["norm_mean"]
#         var = g.ndata["norm_var"]

#         norm_h = (h - mean) / (var + self.eps)

#         if self.affine:
#             return self.gamma * norm_h + self.beta
#         else:
#             return norm_h
