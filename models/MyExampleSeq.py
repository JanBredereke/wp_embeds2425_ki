import brevitas.nn as qnn
import torch.nn as nn

from models.BaseModelSequential import BaseModelSequential


class MyExampleSeq(BaseModelSequential):
    """
    @author <Serkay-Günay Celik>

    Another sequential example model using `BaseModelSequential` with `nn.Sequential`.
    This model is adapted from a previous group implementation.

    Args:
        in_features (int): Number of input features.
        number_of_categories (int): Number of output categories for classification.
    """

    def __init__(self, in_features, number_of_categories):
        """
        Initializes the MyExampleSeq model with a sequence of quantized linear layers,
        batch normalization, dropout, and activation functions.

        :param in_features: Number of input features.
        :param number_of_categories: Number of output classes for classification.
        """
        super().__init__()  # Call the constructor of the BaseModelSequential class

        # Define the model as a sequence of layers using nn.Sequential
        self.model = nn.Sequential(
            # First quantized linear layer with 3-bit precision weights
            qnn.QuantLinear(in_features, in_features * 2, bias=True, weight_bit_width=3),
            # Batch normalization to stabilize training and improve convergence
            nn.BatchNorm1d(in_features * 2),
            # Dropout layer with a 50% dropout rate to reduce overfitting
            nn.Dropout(0.5),
            # Quantized ReLU activation with 4-bit precision
            qnn.QuantReLU(bit_width=4),

            # Second quantized linear layer
            qnn.QuantLinear(in_features * 2, in_features * 2, bias=True, weight_bit_width=3),
            # Batch normalization
            nn.BatchNorm1d(in_features * 2),
            # Dropout layer
            nn.Dropout(0.5),
            # Quantized ReLU activation
            qnn.QuantReLU(bit_width=4),

            # Third quantized linear layer
            qnn.QuantLinear(in_features * 2, in_features * 2, bias=True, weight_bit_width=3),
            # Batch normalization
            nn.BatchNorm1d(in_features * 2),
            # Dropout layer
            nn.Dropout(0.5),
            # Quantized ReLU activation
            qnn.QuantReLU(bit_width=4),

            # Final output layer for classification with quantized linear transformation
            qnn.QuantLinear(in_features * 2, number_of_categories, bias=True, weight_bit_width=3)
        )
