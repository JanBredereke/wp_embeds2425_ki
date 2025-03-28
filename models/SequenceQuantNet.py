import brevitas.nn as qnn
import torch.nn as nn
from brevitas.quant import Int8WeightPerTensorFloat

from models.BaseModelSequential import BaseModelSequential


class SequenceQuantNet(BaseModelSequential):
    """
    @author <Sebastian Schramm>
    Simple quantized neural network with a convolutional layer, an LSTM layer and a fully connected layer.

    Args:
        in_features (int): Number of input features.
        number_of_categories (int): Number of output categories.
        rnn_hidden_size (int): Number of hidden units in the LSTM layer.
    """
    def __init__(self, in_features, number_of_categories, rnn_hidden_size=128):
        super(SequenceQuantNet, self).__init__()
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

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = qnn.QuantLinear(
            in_features=rnn_hidden_size,
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

        x = x.permute(0, 2, 1).contiguous()  # (Batch, Time, Features)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the output from the last time step

        x = self.fc(x)
        return self.softmax(x)
