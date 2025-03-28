import brevitas.nn as qnn  # Import Brevitas' quantized neural network module
import torch.nn as nn  # Import PyTorch's neural network module
from brevitas.quant import Int8WeightPerTensorFloat  # Import quantization method for weights

from models.BaseModelCustom import BaseModelCustom  # Import the base custom model class


class CustomModel(BaseModelCustom):
    """
    @author <Serkay-Günay Celik>

    A custom neural network model extending `BaseModelCustom`.
    This model requires a custom forward method to explicitly define the data flow.

    Args:
        in_features (int): Number of input features.
        number_of_categories (int): Number of output categories (classification classes).
    """

    def __init__(self, in_features, number_of_categories):
        """
        Initializes the CustomModel with quantized linear layers, activation, batch normalization,
        and a final classification layer.

        :param in_features: Number of input features.
        :param number_of_categories: Number of output classes for classification.
        """
        super().__init__()  # Call the constructor of the BaseModelCustom class

        # First quantized linear layer with 3-bit weight precision
        self.layer1 = qnn.QuantLinear(
            in_features,  # Input dimension
            in_features * 2,  # Output dimension (doubling the input size)
            bias=False,  # No bias term
            weight_quant=Int8WeightPerTensorFloat,  # Quantization method for weights
            weight_bit_width=3,  # Weight precision: 3 bits
            return_quant_tensor=True,  # Return a quantized tensor
        )

        # Quantized ReLU activation with 4-bit precision
        self.layer2 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        # Second quantized linear layer, same configuration as the first one
        self.layer3 = qnn.QuantLinear(
            in_features * 2,  # Input dimension (from previous layer)
            in_features * 2,  # Output dimension (same as input)
            weight_quant=Int8WeightPerTensorFloat,  # Quantization method for weights
            bias=False,  # No bias term
            weight_bit_width=3,  # Weight precision: 3 bits
            return_quant_tensor=True,  # Return a quantized tensor
        )

        # Batch normalization layer to stabilize training and improve convergence
        self.layer4 = nn.BatchNorm1d(in_features * 2)

        # Final linear layer for classification
        self.layer5 = nn.Linear(in_features * 2, number_of_categories)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        :param x: Input tensor.
        :return: Output tensor with logits for classification.
        """
        # Pass input through the first quantized linear layer
        x = self.layer1(x)
        # Apply quantized ReLU activation
        x = self.layer2(x)
        # Pass through the second quantized linear layer
        x = self.layer3(x)
        # Apply batch normalization
        x = self.layer4(x)
        # Final output layer for classification
        return self.layer5(x)
