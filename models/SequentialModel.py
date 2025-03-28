import brevitas.nn as qnn
import torch.nn as nn
from brevitas.quant import Int8WeightPerTensorFloat

from models.BaseModelSequential import BaseModelSequential

class SequentialModel(BaseModelSequential):
    """
    @author <Serkay-Günay Celik>

    An example model using `BaseModelSequential` with predefined layers in `nn.Sequential`.
    This model applies quantized linear layers with ReLU activation and batch normalization.

    Args:
        in_features (int): Number of input features.
        number_of_categories (int): Number of output categories for classification.
    """

    def __init__(self, in_features, number_of_categories):
        """
        Initializes the `SequentialModel` with a stack of quantized linear layers,
        ReLU activations, and batch normalization.

        :param in_features: Number of input features.
        :param number_of_categories: Number of output classes for classification.
        """
        super().__init__()  # Call the constructor of the BaseModelSequential class

        # Define the model using nn.Sequential
        self.model = nn.Sequential(
            # First quantized linear layer with 3-bit weight precision
            qnn.QuantLinear(
                in_features,                 # Input feature dimension
                in_features * 2,             # Expands feature space
                bias=False,                  # No bias term
                weight_quant=Int8WeightPerTensorFloat,  # Weight quantization method
                weight_bit_width=3,          # Quantization precision: 3 bits
                return_quant_tensor=True,    # Return quantized tensor
            ),

            # First quantized ReLU activation with 4-bit precision
            qnn.QuantReLU(bit_width=4, return_quant_tensor=True),

            # Second quantized linear layer with the same feature dimension
            qnn.QuantLinear(
                in_features * 2,             # Input from previous layer
                in_features * 2,             # Output feature dimension remains the same
                weight_quant=Int8WeightPerTensorFloat,  # Weight quantization
                bias=False,                  # No bias term
                weight_bit_width=3,          # 3-bit precision for quantized weights
                return_quant_tensor=True,    # Return quantized tensor
            ),

            # Second quantized ReLU activation
            qnn.QuantReLU(bit_width=4, return_quant_tensor=True),

            # Final quantized linear layer mapping to the number of categories
            qnn.QuantLinear(
                in_features * 2,             # Input dimension
                number_of_categories,        # Output matches the number of classes
                weight_quant=Int8WeightPerTensorFloat,  # Weight quantization
                bias=False,                  # No bias term
                weight_bit_width=3,          # 3-bit precision for quantized weights
                return_quant_tensor=False,   # Output is not a quantized tensor
            ),

            # Batch normalization to stabilize the final classification outputs
            nn.BatchNorm1d(number_of_categories),
        )
