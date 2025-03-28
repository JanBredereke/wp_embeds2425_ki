import torch
from torch.nn import Module


class BaseModelSequential(Module):
    """
    @author <Serkay-Günay Celik>

    This class is designed for models with predefined sequential layers.
    The `forward` method processes the input tensor through the layers defined in the `model` attribute.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the model.

        :param x: Input tensor that passes through the model.
        :return: The transformed tensor after being processed by the model's layers.

        Note: `self.model` must be defined in any subclass before calling `forward`,
        otherwise, an AttributeError will be raised.
        """
        return self.model(x)
