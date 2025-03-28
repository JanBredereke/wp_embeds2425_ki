import torch
from torch.nn import Module


class BaseModelCustom(Module):
    """
    @author <Serkay-Günay Celik>

    This class serves as a base class for models requiring a custom `forward` method.
    Subclasses must implement the `forward` method to explicitly define the data flow.
    This is useful for models with non-sequential or complex layer structures.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method that must be overridden by any subclass.

        :param x: Input tensor for the neural network.
        :return: Expected to return an output tensor, but it is not implemented here.
        """
        raise NotImplementedError("You must implement the forward method in the derived class.")
