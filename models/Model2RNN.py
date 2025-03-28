import brevitas.nn as qnn
import torch.nn as nn
from brevitas.quant import Int8WeightPerTensorFloat

from models.BaseModelCustom import BaseModelCustom


class Model2RNN(BaseModelCustom):
    """
    @author <Serkay-Günay Celik>

    An example model that combines a CNN with an RNN (LSTM).
    This model extends `BaseModelCustom` and requires a custom forward method.

    Args:
        in_features (int): Number of input features.
        number_of_categories (int): Number of output categories for classification.
        rnn_hidden_size (int, optional): Size of the hidden state in the LSTM. Default is 128.
    """

    def __init__(self, in_features: int, number_of_categories: int, rnn_hidden_size: int = 128):
        """
        Initializes the Model2RNN with a CNN, LSTM, and a fully connected layer.

        :param in_features: Number of input features.
        :param number_of_categories: Number of output classes for classification.
        :param rnn_hidden_size: Size of the hidden layer in the LSTM.
        """
        super().__init__()  # Call the constructor of the BaseModelCustom class

        # CNN part (feature extraction using quantized linear layers and activation)
        self.layer1 = qnn.QuantLinear(
            in_features=in_features,  # Input dimension
            out_features=in_features * 2,  # Expanding feature representation
            bias=False,  # No bias term
            weight_quant=Int8WeightPerTensorFloat,  # Quantization method for weights
            weight_bit_width=3,  # Weight precision: 3 bits
            return_quant_tensor=True,  # Return a quantized tensor
        )

        # Quantized ReLU activation function with 4-bit precision
        self.layer2 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

        # Second quantized linear layer for feature transformation
        self.layer3 = qnn.QuantLinear(
            in_features=in_features * 2,  # Input dimension (from previous layer)
            out_features=in_features * 2,  # Output dimension remains the same
            weight_quant=Int8WeightPerTensorFloat,  # Quantization method for weights
            bias=False,  # No bias term
            weight_bit_width=3,  # Weight precision: 3 bits
            return_quant_tensor=True,  # Return a quantized tensor
        )

        # Dropout layer to reduce overfitting
        self.layer4 = nn.Dropout(0.2)

        # RNN part (LSTM for sequential feature learning)
        self.rnn = nn.LSTM(
            input_size=in_features * 2,  # Input dimension must match CNN output
            hidden_size=rnn_hidden_size,  # Hidden state size of the LSTM
            num_layers=1,  # Single LSTM layer
            batch_first=True  # Ensure batch is the first dimension
        )

        # Fully connected layer for classification
        self.layer5 = qnn.QuantLinear(
            in_features=rnn_hidden_size,  # Input from LSTM hidden state
            out_features=number_of_categories,  # Number of output classes
            weight_quant=Int8WeightPerTensorFloat,  # Quantization method
            bias=False,  # No bias term
            weight_bit_width=3,  # Weight precision: 3 bits
            return_quant_tensor=False,  # Output is not a quantized tensor
        )

        # Batch normalization to improve stability of the output
        self.norm = nn.BatchNorm1d(number_of_categories)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        :param x: Input tensor of shape (Batch, Features, Time).
        :return: Output tensor with classification logits.
        """

        # CNN part (Feature extraction)
        x = self.layer1(x)  # First quantized linear transformation
        x = self.layer2(x)  # Quantized ReLU activation
        x = self.layer3(x)  # Second quantized linear transformation
        x = self.layer4(x)  # Apply dropout to prevent overfitting

        # Transform input for RNN: Change shape to (Batch, Time, Features)
        x = x.permute(0, 2, 1).contiguous()  # Ensure memory continuity for RNN

        # RNN part (LSTM processing)
        x, _ = self.rnn(x)  # Pass through the LSTM network

        # Take the output from the last time step
        x = x[:, -1, :]  # Extract the last time step's output from LSTM

        # Fully connected output layer for classification
        x = self.layer5(x)

        # Apply batch normalization to stabilize the final output
        return self.norm(x)
