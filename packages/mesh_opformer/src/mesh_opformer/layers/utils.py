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


class AccelerateOperator(nn.Module):
    """ """

    def __init__(self, dim: int) -> None:
        """
        Parameters
        ----------
        dim : int
            Input dimensionality.
        """
        super(AccelerateOperator, self).__init__()
        self.dim = dim
        self.diffusion_time = nn.Parameter(torch.Tensor(dim))

        nn.init.constant_(self.diffusion_time, 0.0)

    def forward(self, x_spec: torch.Tensor) -> torch.Tensor:
        """
        Apply accelerate operator.
        """
        # Constrain parameter to positive halfspace, and away from 0
        # in the incredibly rare chance that they get stuck.
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        if x_spec.shape[-1] != self.dim:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x_spec.shape, self.dim
                )
            )

        diffusion_coefs = torch.exp(self.diffusion_time.unsqueeze(0))

        return diffusion_coefs * x_spec
