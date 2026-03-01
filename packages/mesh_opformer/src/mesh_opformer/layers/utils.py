import copy
import torch

from torch import nn


def clones(module, N):
    """
    Produce N identical layers.

    Parameters
    ----------
    module : nn.Module
        An `nn.Module` instance to be copied.
    N : int
        Number of copies.

    Returns
    -------
    nn.ModuleList
        A list of identical nn.Modules.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def validate_modes(modes):
    """
    Validate input modes.

    Parameters
    ----------
    modes : Tuple[str]
        The mesh elements 'v', 'e' or 'f' on which attention operations
        will be defined. Any combination of these flags is valid, e.g.
        "ve" creates learnable layers for nodes and edges.

    Returns
    -------
    Tuple[str]
        Validate input modes.

    Raises
    ------
    ValueError
        If modes are not valid.
    """
    MODES = ("v", "e", "f")

    modes = tuple(set(modes))

    if not modes:
        raise ValueError("Input modes are empty. Choose from {}".format(MODES))

    if not set(modes).issubset(MODES):
        raise ValueError("Input modes are not valid. Choose from {}".format(MODES))

    return modes


class NoOperation(torch.nn.Module):
    """
    A class representing a No Operation.

    Similar to `nn.Identity` but with support for multiple
    `forward` inputs.
    """

    def __init__(self, *args: any, **kwargs: any) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, *args: any, **kwargs: any) -> torch.Tensor:
        """
        The forward method returns the first input and ignores the rest.
        """
        return input
