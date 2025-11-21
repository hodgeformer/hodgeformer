"""
Since no softmax layer is applied, computations can be carried out efficiently 
from right to left and avoid making intermediate matrices of dimensions, (nxn),
(mxn), (mxm).

By carrying computations from right to left, no explicit attention matrix.
"""

import torch
import math


def linear_attention__flowformer(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Apply flowformer linear attention, similar to Wu et al. (2022):
    `Flowformer: Linearizing Transformers with Conservation Flows`

    Implementation based on paper.
    """
    q = torch.nn.functional.sigmoid(q)
    k = torch.nn.functional.sigmoid(k)

    I = (q + eps) * (k.sum(dim=-2, keepdim=True) + eps)
    I = I.sum(dim=-1, keepdim=True)

    O = (k + eps) * (q.sum(dim=-2, keepdim=True) + eps)
    O = O.sum(dim=-1, keepdim=True)

    I_hat = (q + eps) * (torch.div(k, O).sum(dim=-2, keepdim=True) + eps)
    I_hat = I_hat.sum(dim=-1, keepdim=True)

    O_hat = (k + eps) * (torch.div(q, I).sum(dim=-2, keepdim=True) + eps)
    O_hat = O_hat.sum(dim=-1, keepdim=True)

    v = v * O_hat.softmax(dim=-1)

    # K.T @ x
    v = torch.einsum("bhnd,bhne->bhde", k, v)
    # Q @ (K.T @ x)
    v = torch.einsum("bhnd,bhde->bhne", torch.div(q, I), v)

    R = v * torch.nn.functional.sigmoid(I_hat)

    return R


def linear_attention__efficient(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """
    Apply efficient linear attention, similar to Shen et al. (2020):
    `Efficient Attention: Attention with Linear Complexities`

    Follow implementation from:

    https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py#L204
    """
    d_q = q.size(-1)

    # Row normalize q values, column normalize k values
    q = q.softmax(dim=-1)
    k = k.softmax(dim=-2)

    # normalize q
    q = q / math.sqrt(d_q)

    # KV = K.T @ V
    kv = torch.einsum("bhnd,bhne->bhde", k, v)
    # Q @ KV
    x = torch.einsum("bhnd,bhde->bhne", q, kv)

    return x


def linear_attention__fast_transformers(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Apply linear attention, similar to (Katharopoulos et al., 2020):
    `Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention`

    Follow implementation from:

    https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py#L72
    """
    q = torch.nn.functional.elu(q) + 1
    k = torch.nn.functional.elu(k) + 1

    # KV = K.T @ V
    kv = torch.einsum("bhkd,bhke->bhde", k, v)
    # normalization
    z = 1.0 / (torch.einsum("bhkd,bhd->bhk", q, k.sum(dim=-2)) + eps)
    # Q @ KV
    x = torch.einsum("bhkd,bhde,bhk->bhke", q, kv, z)

    return x


def liner_attention__sgformer(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Follow implmentation from:

    https://github.com/qitianwu/SGFormer/blob/main/medium/difformer.py#L18C1-L43C96
    """
    n, d_q = q.size(-2), q.size(-1)

    # normalize input
    q = q / torch.linalg.norm(q, p=2)  # [N, H, M]
    k = k / torch.linalg.norm(k, p=2)  # [L, H, M]

    # # numerator
    # kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    # attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs) # [N, H, D]
    # all_ones = torch.ones([vs.shape[0]]).to(vs.device)
    # vs_sum = torch.einsum("l,lhd->hd", all_ones, vs) # [H, D]
    # attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1) # [N, H, D]

    # # denominator
    # all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    # ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    # attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # # attentive aggregated results
    # attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    # attention_normalizer += torch.ones_like(attention_normalizer) * N
    # attn_output = attention_num / attention_normalizer # [N, H, D]

    # # compute attention for visualization if needed
    # if output_attn:
    #     attention = torch.einsum("nhm,lhm->nlh", qs, ks) / attention_normalizer # [N, L, H]
    pass
