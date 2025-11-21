import torch
import torch.nn as nn


class MLPSimple(nn.Sequential):
    """
    A simple MLP with configurable hidden layer sizes.
    """

    def __init__(
        self, layer_sizes, dropout=None, activation: str = "relu", name="mlp_simple"
    ):
        """
        Parameters
        ----------
        layer_sizes : Iterable[int]
            A list of integers indicating size of each hidden layer.
        dropout : float | None
            Dropout probability for each layer.
        activation : {'relu', 'gelu', 'silu'}
            Type of activation for MLP layers.
        name : str
            Name of the layer.
        """
        super(MLPSimple, self).__init__()

        activations = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }

        activation_cls = activations[activation]

        for i in range(len(layer_sizes) - 1):

            # Linear layer (Affine map)
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
            )

            if dropout:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i), nn.Dropout(p=dropout)
                )

            self.add_module(name + "_mlp_act_{:03d}".format(i), activation_cls())


class MLPTwoLayer(nn.Module):
    """
    Class to represent a 2-layer MLP neural network.
    """

    def __init__(
        self,
        in_dim: int = 256,
        hid_dim: int = 1024,
        out_dim: int | None = None,
        batch_norm: bool = False,
        activation: str = "relu",
        dropout: float | None = 0.1,
    ) -> None:
        """
        Initialization for `MLPTwoLayer` class.

        Parameters
        ----------
        in_dim : int
            Input dimension for first linear layer.
        hid_dim : int
            Output dimension for first linear layer.
        out_dim : int | None
            Output dimension for second linear layer. If not specified
            the `in_dim` is used.
        batch_norm : bool
            Whether to use batch normalization between the linear layers.
        activation : {'relu', 'silu', 'gelu'}
            Activation functions.
        dropout : float | None
            Dropout probability for between layers.
        """
        super(MLPTwoLayer, self).__init__()

        activations = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }

        out_dim = out_dim if out_dim is not None else in_dim

        self.linear_1 = nn.Linear(in_dim, hid_dim)
        self.linear_2 = nn.Linear(hid_dim, out_dim)

        self.activation = activations[activation]()

        self.use_batch_norm = batch_norm
        self.use_dropout = bool(dropout)

        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hid_dim)

        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.activation(self.linear_1(x))

        if self.use_dropout:
            x = self.dropout(x)

        if self.use_batch_norm:
            # Permute dimensions to apply batch norm?
            x = x.permute((0, 2, 1))
            x = self.batch_norm(x)
            x = x.permute((0, 2, 1))

        x = self.linear_2(x)

        return x
