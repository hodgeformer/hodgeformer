from typing import Dict

import copy

from torch import nn

from .encoder import MeshHodgeFormerLayer, MeshHodgeFormerEncoder
from .attention_hodge_laplace import MeshMultiHeadHodgeLaplaceAttention
from .attention_hodge_dirac import MeshMultiHeadHodgeDiracAttention

from .transformer import init_layer__transformer
from .fc import MLPTwoLayer
from .embedding import LinearEmbedding, NeighborEmbedding
from .tasks import MeshClassifier, MeshSegmenter, MeshFunctionalMapper

from .utils import clones, validate_modes


def build_mesh_opformer_model(
    v_in: int,
    e_in: int,
    f_in: int,
    d_v: int = 128,
    d_e: int = 128,
    d_f: int = 128,
    d_hidden: int = 256,
    N: int = 4,
    dropout: float = 0.1,
    modes: str = "ve",
    embed_type: str = "linear",
    attn_type: str = "hodge",
    task_type: str = "classification",
    layer_layout: str = "2:1e",
    embed_kw: Dict = {},
    attn_kw: Dict = {},
    attn_basic_kw: Dict = {},
    task_kw: Dict = {},
    # h: int = 8,
    # attn_sym: bool = True,
    # attn_norm: bool = False,
    # hodge_diag: bool = True,
    # attn_acc: bool = True,
    # attn_mask: bool = False,
    # dropout_attn: float | None = None,
) -> nn.Module:
    """
    Builder function for a Mesh OpFormer model.

    Parameters
    ----------
    v_in : int
        Input dimensionality of vertex features.
    e_in : int
        Input dimensionality of edge features.
    f_in : int
        Input dimensionality of face features.
    d_v : int
        Dimensionality of vertex embeddings.
    d_e : int
        Dimensionality of edge embeddings.
    d_f : int
        Dimensionality of face embeddings.
    d_hidden : int
        Dimensionality of feed-forward hidden layer.
    N : int
        Number of layers.
    dropout : float
        Dropout probability.
    modes : str
        The mesh elements 'v', 'e' or 'f' on which attention operations
        will be defined. Any combination of these flags is valid, e.g.
        "ve" creates learnable layers for nodes and edges.
    embed_type : {'linear', 'neighbor'}
        The type of input embedding layer to apply.
    attn_type : {'hodge', 'simple'}
        Type of attention layer to use: The 'hodge' attention layer
        estimates hodge operators with attention. The 'simple' attention
        operator.
    task_type : {'classification', 'encode'}
        The task head to add to the transformer layer.
    embed_kw : Dict
        Keyword arguments to pass to the embedding layer.
    attn_kw : Dict
        Keyword arguments to pass to the Hodge attention layer.
    attn_basic_kw : Dict
        Keyword arguments to pass to the Basic attention layer.
    task_kw : Dict
        Keyword arguments to pass to the task head layer.

    Returns
    -------
    nn.Module
        A Mesh Operator Transformer model.
    """
    modes = validate_modes(modes)

    EMBEDDING_LAYERS = {
        "linear": LinearEmbedding,
        "neighbor": NeighborEmbedding,
    }

    embedding_layer = EMBEDDING_LAYERS[embed_type](
        v_in=v_in, e_in=e_in, f_in=f_in, d_v=d_v, d_e=d_e, d_f=d_f, **embed_kw
    )

    hodge_laplace_layer = init_layer__hodgeformer(
        d_v=d_v,
        d_e=d_e,
        d_f=d_f,
        d_hidden=d_hidden,
        dropout=dropout,
        modes=modes,
        attn_type="hodge",
        attn_kw=attn_kw,
    )

    hodge_dirac_layer = init_layer__hodgeformer(
        d_v=d_v,
        d_e=d_e,
        d_f=d_f,
        d_hidden=d_hidden,
        dropout=dropout,
        modes=modes,
        attn_type="dirac",
        attn_kw=attn_kw,
    )

    transformer_layer = init_layer__transformer(
        d=d_v, d_hidden=d_hidden, dropout=dropout, attn_kw=attn_basic_kw
    )

    encoder = MeshHodgeFormerEncoder(
        embed_layer=embedding_layer,
        laplace_layer=hodge_laplace_layer,
        dirac_layer=hodge_dirac_layer,
        basic_layer=transformer_layer,
        N=N,
        modes=modes,
        layer_layout=layer_layout,
    )

    if task_type == "encode":
        model = encoder

    elif task_type == "classification":
        model = MeshClassifier(encoder=encoder, d_emb=d_v, **task_kw)

    elif task_type == "segmentation":
        model = MeshSegmenter(encoder=encoder, d_emb=d_v, **task_kw)

    elif task_type == "functional-correspondence":
        model = MeshFunctionalMapper(encoder=encoder, **task_kw)

    else:
        raise ValueError("No other task is supported yet.")

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def init_layer__hodgeformer(
    d_v: int,
    d_e: int,
    d_f: int,
    d_hidden: int,
    dropout: float,
    modes: str,
    attn_type: str,
    attn_kw: Dict,
):
    """
    Initialize a `MeshHodgeFormerLayer`.
    """
    ATTN_LAYERS = {
        "hodge": MeshMultiHeadHodgeLaplaceAttention,
        "dirac": MeshMultiHeadHodgeDiracAttention,
    }

    attn_layer_cls = ATTN_LAYERS[attn_type]

    mesh_attn = attn_layer_cls(
        d_v=d_v,
        d_e=d_e,
        d_f=d_f,
        dropout=dropout,
        modes=modes,
        **attn_kw,
        # h=h,
        # dropout_attn=dropout_attn,
        # attn_sym=attn_sym,
        # attn_norm=attn_norm,
        # attn_mask=attn_mask,
        # hodge_diag=hodge_diag,
        # attn_acc=attn_acc,
    )

    mlp_kw = {"batch_norm": True, "activation": "relu", "dropout": dropout}

    v_ff = MLPTwoLayer(d_v, d_hidden, None, **mlp_kw) if "v" in modes else nn.Identity()
    e_ff = MLPTwoLayer(d_e, d_hidden, None, **mlp_kw) if "e" in modes else nn.Identity()
    f_ff = MLPTwoLayer(d_f, d_hidden, None, **mlp_kw) if "f" in modes else nn.Identity()

    mesh_hodgeformer_module = MeshHodgeFormerLayer(
        d_v=d_v,
        d_e=d_e,
        d_f=d_f,
        mesh_attn=copy.deepcopy(mesh_attn),
        v_ff=copy.deepcopy(v_ff),
        e_ff=copy.deepcopy(e_ff),
        f_ff=copy.deepcopy(f_ff),
        dropout=dropout,
        modes=modes,
    )

    return mesh_hodgeformer_module
