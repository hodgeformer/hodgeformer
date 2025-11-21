from typing import Dict

import itertools
import torch

from torch import nn


def _l1_regularization(params: torch.Tensor, _lambda: float = 0.1) -> torch.Tensor:
    """
    Add L1 regularization to attention.
    """
    loss_reg = 0

    for param in params:
        loss_reg += _lambda * torch.norm(param, 1)

    return loss_reg


def _l2_regularization(params: torch.Tensor, _lambda: float = 0.1) -> torch.Tensor:
    """
    Add L2 regularization to attention.
    """
    loss_reg = 0

    for param in params:
        loss_reg += _lambda * torch.norm(param, 2) ** 2

    return loss_reg


def _graph_trend_filtering(
    activations: Dict[str, torch.Tensor], d_0: torch.Tensor, _lambda: float = 0.001
) -> torch.Tensor:
    """
    Add graph-trend-filtering.
    """
    loss = 0

    for _, activation in activations.items():

        _tmp = torch.bmm(d_0, activation)
        _tmp = torch.bmm(d_0.transpose(-2, -1), _tmp)
        _tmp = torch.bmm(activation.transpose(-2, -1), _tmp)

        loss += torch.einsum("...ii", _tmp).mean()

        # norm = torch.norm(_tmp, 1, dim=(1, 2))
        # loss += torch.mean(norm)

    loss *= _lambda

    return loss


def get_attention_activations(model: nn.Module) -> Dict[str, Dict]:
    """
    Regularize attention Q, K activation layers.
    """
    OUTPUTS = {}

    def get_activation_hook(space, name):
        """
        Get activations to regularize via forward hook.
        """

        def hook(model, input, output):

            if space not in OUTPUTS:
                OUTPUTS[space] = {}

            OUTPUTS[space][name] = output

            return output

        return hook

    enc_layers = model.get_submodule("encoder.layers")

    N = len(enc_layers)

    SUBMODULE = "{}.mesh_attn.{}.{}"

    LAYERS_IDX = [i for i in range(N)]
    SPACES_ATTN = ["v_linears", "e_linears", "f_linears"]
    LAYERS_Q_K_V = [0]

    for layer_idx, space, linear_idx in itertools.product(
        LAYERS_IDX, SPACES_ATTN, LAYERS_Q_K_V
    ):
        submodule = SUBMODULE.format(layer_idx, space, linear_idx)

        layer = enc_layers.get_submodule(submodule)

        layer.register_forward_hook(get_activation_hook(space, submodule))

    return OUTPUTS


def regularize_attention_weights(
    model: nn.Module, regularization: str = "l1", _lambda: float = 0.1
) -> torch.Tensor:
    """
    Add regularization to attention linear layers.
    """
    enc_layers = model.get_submodule("encoder.layers")

    N = len(enc_layers)

    SUBMODULE = "{}.mesh_attn.{}.{}"

    LAYERS_IDX = [i for i in range(N)]
    LAYERS_ATTN = ["v_linears", "e_linears", "f_linears"]
    LAYERS_Q_K_V = [0, 1]

    # Gather Q, K parameters from attention layers
    params = []

    for layer_idx, space, linear_idx in itertools.product(
        LAYERS_IDX, LAYERS_ATTN, LAYERS_Q_K_V
    ):

        submodule = SUBMODULE.format(layer_idx, space, linear_idx)

        layer = enc_layers.get_submodule(submodule)

        params.extend([param for param in layer.parameters()])

    if regularization == "l1":
        return _l1_regularization(params, _lambda)

    elif regularization == "l2":
        return _l2_regularization(params, _lambda)
