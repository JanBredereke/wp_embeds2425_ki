import brevitas.nn as qnn
import torch.nn as nn
from brevitas.quant import Int8WeightPerTensorFloat

from models.BaseModelSequential import BaseModelSequential


class PartiallyQuantizedNet(BaseModelSequential):
    """
    @author <Sebastian Schramm>
    Simple quantized neural network with two convolutional layers and two pooling layers.

    Args:
        in_features (int): Number of input features.
        number_of_categories (int): Number of output categories.
    """
    def __init__(self, in_features, number_of_categories):
        super(PartiallyQuantizedNet, self).__init__()
        self.conv1 = qnn.QuantConv1d(
            in_channels=in_features,
            out_channels=32,
            kernel_size=5,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=3,
            return_quant_tensor=False
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv2 = qnn.QuantConv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=3,
            return_quant_tensor=False
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.fc = qnn.QuantLinear(
            in_features=64 * 10,  # Adjust based on input size and pooling
            out_features=number_of_categories,
            weight_quant=Int8WeightPerTensorFloat,
            bias=False,
            weight_bit_width=3,
            return_quant_tensor=False
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return self.softmax(x)
