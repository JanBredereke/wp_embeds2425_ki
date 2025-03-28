import brevitas.nn as qnn
import torch.nn as nn
from brevitas.quant import Int8WeightPerTensorFloat

from models.BaseModelSequential import BaseModelSequential


class ModerateQuantNet(BaseModelSequential):
    """
    @author <Sebastian Schramm>
    Simple quantized neural network with three convolutional layers and three pooling layers.

    Args:
        in_features (int): Number of input features.
        number_of_categories (int): Number of output categories.
    """
    def __init__(self, in_features, number_of_categories):
        super(ModerateQuantNet, self).__init__()
        self.conv1 = qnn.QuantConv1d(
            in_channels=in_features,
            out_channels=32,
            kernel_size=5,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=3,
            return_quant_tensor=True
        )
        self.relu1 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.pool1 = qnn.QuantAvgPool1d(kernel_size=2, stride=2)

        self.conv2 = qnn.QuantConv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=3,
            return_quant_tensor=True
        )
        self.relu2 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.pool2 = qnn.QuantAvgPool1d(kernel_size=2, stride=2)

        self.conv3 = qnn.QuantConv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=3,
            return_quant_tensor=True
        )
        self.relu3 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.pool3 = qnn.QuantAvgPool1d(kernel_size=2, stride=2)

        self.fc = qnn.QuantLinear(
            in_features=128 * 10,  # Adjust based on input size and pooling
            out_features=number_of_categories,
            weight_quant=Int8WeightPerTensorFloat,
            bias=False,
            weight_bit_width=3,
            return_quant_tensor=False
        )
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = self.dropout(x)
        return self.softmax(x)
