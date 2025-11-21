from typing import Tuple, List

import os
import json

import torch
import numpy as np

from scipy.sparse import spmatrix


def _from_json(json_path: str, data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Get mesh paths & labels from `.json` file.
    """
    pass


def _from_dir(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Get mesh paths & labels from path to root directory.
    """
    paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    labels = [0 if "benign" in path else 1 for path in paths]  # ?

    return paths, labels


def _from_csv(csv_path: str, data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Get mesh paths & labels from `.csv` file.
    """
    usecols = ["name", "label"]

    df = pd.read_csv(csv_path, usecols=usecols)  # type: ignore

    paths = (data_dir + df["name"]).to_list()
    labels = df["label"].to_list()

    return paths, labels


def sparse_scipy_to_torch(
    scp: spmatrix, transpose: bool = False, dtype: torch.dtype = None
) -> torch.sparse.FloatTensor:
    """
    Convert scipy sparse matrix to torch sparse tensor.
    """
    coo = scp.tocoo()

    if transpose is True:
        indices = np.vstack((coo.col, coo.row))
        size = tuple(reversed(coo.shape))
    else:
        indices = np.vstack((coo.row, coo.col))
        size = coo.shape

    values = coo.data.astype("float32")

    return torch.sparse_coo_tensor(
        indices=indices, values=values, size=size, is_coalesced=True, dtype=dtype
    )


# `torch` does not support sparse matrices throught its default collate fn
def collate_batch_w_sparse_fn(batch):
    """
    Collate function to handle sparse tensors in batch.
    """
    # Handle mesh features
    x_vs = _collate_tensor_w_pad([elem[0][0] for elem in batch])
    x_es = _collate_tensor_w_pad([elem[0][1] for elem in batch])
    x_fs = _collate_tensor_w_pad([elem[0][2] for elem in batch])
    d_0s = _collate_sparse_w_pad([elem[0][3] for elem in batch])
    d_1s = _collate_sparse_w_pad([elem[0][4] for elem in batch])

    v_idx = _collate_tensor_w_pad([elem[0][5] for elem in batch])
    e_idx = _collate_tensor_w_pad([elem[0][6] for elem in batch])
    f_idx = _collate_tensor_w_pad([elem[0][7] for elem in batch])

    # Handle labels
    labels = _collate_tensor_w_pad([elem[1] for elem in batch], padding_value=-1)

    # Handle class weights
    weights = _collate_tensor([elem[2] for elem in batch])

    return (x_vs, x_es, x_fs, d_0s, d_1s, v_idx, e_idx, f_idx), labels, weights


# `torch` does not support sparse matrices throught its default collate fn
def collate_functional_corr_batch_w_sparse_fn(batch):
    """
    Collate function to handle sparse tensors in batch.
    """
    # Handle mesh features
    x_vs_s = _collate_tensor_w_pad([elem[0][0] for elem in batch])
    x_es_s = _collate_tensor_w_pad([elem[0][1] for elem in batch])
    x_fs_s = _collate_tensor_w_pad([elem[0][2] for elem in batch])
    d_0s_s = _collate_sparse_w_pad([elem[0][3] for elem in batch])
    d_1s_s = _collate_sparse_w_pad([elem[0][4] for elem in batch])

    v_idx_s = _collate_tensor_w_pad([elem[0][5] for elem in batch])
    e_idx_s = _collate_tensor_w_pad([elem[0][6] for elem in batch])
    f_idx_s = _collate_tensor_w_pad([elem[0][7] for elem in batch])

    x_vs_t = _collate_tensor_w_pad([elem[1][0] for elem in batch])
    x_es_t = _collate_tensor_w_pad([elem[1][1] for elem in batch])
    x_fs_t = _collate_tensor_w_pad([elem[1][2] for elem in batch])
    d_0s_t = _collate_sparse_w_pad([elem[1][3] for elem in batch])
    d_1s_t = _collate_sparse_w_pad([elem[1][4] for elem in batch])

    v_idx_t = _collate_tensor_w_pad([elem[1][5] for elem in batch])
    e_idx_t = _collate_tensor_w_pad([elem[1][6] for elem in batch])
    f_idx_t = _collate_tensor_w_pad([elem[1][7] for elem in batch])

    # Handle ground truths
    gts = _collate_tensor_w_pad([elem[2] for elem in batch], padding_value=-1)

    evecs_s = _collate_tensor_w_pad([elem[3][0] for elem in batch], padding_value=-1)
    evals_s = _collate_tensor_w_pad([elem[3][1] for elem in batch], padding_value=-1)
    vts_s = _collate_tensor_w_pad([elem[3][2] for elem in batch], padding_value=-1)

    evecs_t = _collate_tensor_w_pad([elem[4][0] for elem in batch], padding_value=-1)
    evals_t = _collate_tensor_w_pad([elem[4][1] for elem in batch], padding_value=-1)
    vts_t = _collate_tensor_w_pad([elem[4][2] for elem in batch], padding_value=-1)

    i_idx = _collate_tensor_w_pad([elem[5] for elem in batch], padding_value=-1)

    return (
        (
            x_vs_s,
            x_es_s,
            x_fs_s,
            d_0s_s,
            d_1s_s,
            v_idx_s,
            e_idx_s,
            f_idx_s,
        ),
        (
            x_vs_t,
            x_es_t,
            x_fs_t,
            d_0s_t,
            d_1s_t,
            v_idx_t,
            e_idx_t,
            f_idx_t,
        ),
        gts,
        (evecs_s, evals_s, vts_s),
        (evecs_t, evals_t, vts_t),
        i_idx
    )


def collate_batch_w_sparse_fn_infer(batch):
    """
    Collate function to handle sparse tensors in batch.
    """
    # Handle mesh features
    x_vs = _collate_tensor_w_pad([elem[0] for elem in batch])
    x_es = _collate_tensor_w_pad([elem[1] for elem in batch])
    x_fs = _collate_tensor_w_pad([elem[2] for elem in batch])
    d_0s = _collate_sparse_w_pad([elem[3] for elem in batch])
    d_1s = _collate_sparse_w_pad([elem[4] for elem in batch])

    v_idx = _collate_tensor_w_pad([elem[5] for elem in batch])
    e_idx = _collate_tensor_w_pad([elem[6] for elem in batch])
    f_idx = _collate_tensor_w_pad([elem[7] for elem in batch])

    return (x_vs, x_es, x_fs, d_0s, d_1s, v_idx, e_idx, f_idx)


def _collate_tensor(elems: List[torch.Tensor]) -> torch.Tensor:
    """
    Collate tensors.
    """
    elem = elems[0]
    out = None

    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in elems)
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(len(elems), *list(elem.size()))

    return torch.stack(elems, 0, out=out)


def _collate_sparse(elems: List[torch.Tensor]) -> torch.Tensor:
    """
    Collate sparse tensors.
    """
    # Concat 2D coo sparse tensors to a 3D sparse coo tensor
    return torch.stack([sparse_elem for sparse_elem in elems])


def _collate_tensor_w_pad(elems: List[torch.Tensor], **kw) -> torch.Tensor:
    """
    Collate uneven tensors to maximum length.
    """
    # elem = max(elems, key=lambda x: x.size(0))
    # out = None

    # if torch.utils.data.get_worker_info() is not None:
    #     # If we're in a background process, concatenate directly into a
    #     # shared memory tensor to avoid an extra copy
    #     numel = sum(elem.numel() for i in range(len(elems)))
    #     storage = elem._typed_storage()._new_shared(numel, device=elem.device)
    #     out = elem.new(storage).resize_(len(elems), *list(elem.size()))
    return torch.nn.utils.rnn.pad_sequence(elems, batch_first=True, **kw)


def _collate_sparse_w_pad(elems: List[torch.Tensor]) -> torch.Tensor:
    """
    Collate sparse tensors to maximum length.
    """
    # Concat 2D coo sparse tensors to a 3D sparse coo tensor
    r_max = max(elem.size(0) for elem in elems)
    c_max = max(elem.size(1) for elem in elems)

    nnz_max = max(elem._nnz() for elem in elems)

    _elems = []

    for elem in elems:

        nnz = elem._nnz()
        size = elem.size()

        if nnz == nnz_max and size == (r_max, c_max):
            _elems.append(elem)
            continue

        if nnz < nnz_max:
            indices = torch.nn.functional.pad(elem.indices(), (0, nnz_max - nnz))
            values = torch.nn.functional.pad(elem.values(), (0, nnz_max - nnz))
        else:
            indices = elem.indices()
            values = elem.values()

        _elems.append(
            torch.sparse_coo_tensor(indices=indices, values=values, size=(r_max, c_max))
        )

    return torch.stack(_elems)


def parse_classification_label(label: int) -> torch.Tensor:
    """
    Parse labels for the mesh classification task.
    """
    if isinstance(label, int):
        label = [label]

    return torch.Tensor(label).to(torch.int64)


def parse_segmentation_label(label_path: str) -> torch.Tensor:
    """
    Parse labels for the mesh segmentation task.
    """
    labels = np.loadtxt(label_path).astype("int32") - 1

    return torch.tensor(np.ascontiguousarray(labels), dtype=torch.long)


def parse_segmentation_label_coseg(label_path: str) -> torch.Tensor:
    """
    Parse labels for the mesh segmentation task.
    """
    labels = np.loadtxt(label_path).astype("int32")

    return torch.tensor(np.ascontiguousarray(labels), dtype=torch.long)


def parse_segmentation_human_simplified_label(label_path: str) -> torch.Tensor:
    """
    Parse labels for the mesh segmentation task.
    """
    labels = np.loadtxt(label_path).astype("int32")

    return torch.tensor(np.ascontiguousarray(labels), dtype=torch.long)


def parse_segmentation_rna_label(label_path: str) -> torch.Tensor:
    """
    Parse labels for the rna-surface segmentation dataset.
    """
    labels = np.loadtxt(label_path).astype("int32") + 1

    return torch.tensor(np.ascontiguousarray(labels), dtype=torch.long)
