import torch


def expand_dim(t: torch.Tensor, dim: int, k: int) -> torch.Tensor:
    """
    Expand dimension.
    """
    # Add dimension at `dim`
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def prepare_indices(idx, h, idx_global=None):
    """
    Prepare indices for gather operation.
    """
    # idx is of shape (b, n, d)
    # Convert global node rows to -1 since the
    # global node columns are already covered
    if idx_global is not None:
        idx.index_fill_(1, idx_global, -1)

    # Handle `-1` values
    mask = idx == -1
    idx = idx.masked_fill(mask, 0)

    # Expand across the `head` dimension
    idx = expand_dim(idx, 1, h)

    idx = idx.reshape(*idx.size()[:2], -1)

    return idx, mask


def batched_index_select(vals, idx, d):
    """
    Select batched indices.
    """
    return vals.gather(2, expand_dim(idx, -1, d))


def scatter_mean(src, t, index, dim, eps=1e-5):
    """
    Scatter mean.

    Parameters
    ----------
    src : torch.Tensor
        An source tensor to add values on.
    t: torch.Tensor
        An input tensor to scatter in different indices.
    index : torch.Tensor
        An index tensor.
    eps : float
        Small float for numerical stability

    Returns
    -------
    torch.Tensor
        A return
    """
    numer = src.scatter_add(dim, index, t)
    denom = src.scatter_add(dim, index, torch.ones_like(t))
    return numer / (denom + eps)


def normalize_w_trace_from_adj_arr(
    adj_arr: torch.Tensor, adj_vals: torch.Tensor, mask_value: int = -1
) -> torch.Tensor:
    """
    Estimate operator trace from array with adjacency indices.
    """
    _eps = 1e-5

    _, h, n, d = adj_vals.size()

    row_idx = torch.repeat_interleave(torch.arange(n), d).to(adj_arr.device.type)
    col_idx = adj_arr.flatten(start_dim=1)

    trace_mask = (row_idx == col_idx) & (col_idx != mask_value)

    trace_mask = expand_dim(trace_mask, 1, h)

    trace = (trace_mask * adj_vals.flatten(start_dim=2)).sum(dim=-1, keepdim=True)

    return adj_vals / (trace.unsqueeze(-1) + _eps)


def sparse_attention(x, q, k, idx, dropout=None, idx_global=None):
    """
    Perform Sparse Attention given a sparsity pattern for each input entry.
    """
    _q, _k = q, k

    b, h, n, d = _q.size()

    # idx_global = torch.LongTensor([0, 1, 2, 3]).to(idx.device.type)

    if idx_global is not None:
        x_global = _global_attention_across_indices(_q, _k, x, dropout, idx_global)
        m_global = _global_mask(x, idx_global)

    # For now do not fill global with -1s as they are overwritten below
    with torch.no_grad():
        _idx, mask = prepare_indices(idx.detach(), h, None)

    _k = batched_index_select(_k, _idx, d)
    _v = batched_index_select(x, _idx, d)

    _k = _k.reshape(b, h, n, -1, d)
    _v = _v.reshape(b, h, n, -1, d)

    qk = torch.einsum("bhnd,bhnjd->bhnj", _q, _k) * (d**-0.5)

    mask = expand_dim(mask, 1, h)

    # Galerkin style normalization
    # qk = qk.masked_fill_(mask, 0.0)
    # qk = torch.nn.functional.normalize(qk, p=1.0, dim=-1)

    qk = qk.masked_fill_(mask, float("-inf"))
    qk = torch.nn.functional.softmax(qk, dim=-1)

    if dropout is not None:
        qk = torch.nn.functional.dropout(qk, p=dropout)

    _v = torch.einsum("bhnj,bhnjd->bhnd", qk, _v)

    if idx_global is not None:
        _v = _v.masked_scatter_(m_global, x_global)

    return _v


def _global_attention_across_indices(q, k, x, dropout, idx_global):
    """
    Calculate global attention across indices.
    """
    _, _, _, d = q.size()

    q_global = q.index_select(2, idx_global)

    qk_global = torch.einsum("bhnd,bhmd->bhnm", q_global, k) * (d**-0.5)
    qk_global = torch.nn.functional.softmax(qk_global, dim=-1)

    if dropout is not None:
        qk_global = torch.nn.functional.dropout(qk_global, p=dropout)

    return torch.einsum("bhlk,bhkd->bhld", qk_global, x)


def _global_mask(x, idx_global):
    """
    Create global mask for scatter assign operation.
    """
    with torch.no_grad():
        m_global = torch.zeros_like(x, dtype=torch.bool)
        m_global[:, :, idx_global, :] = True

    return m_global
