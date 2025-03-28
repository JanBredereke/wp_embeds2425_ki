import brevitas.nn as qnn
import torch


class AudioNet(torch.nn.Module):
    """
    @author <Sebastian Schramm>
    Simple quantized neural network with three convolutional layers and three pooling layers.

    Args:
        in_features (int): Number of input features.
        number_of_categories (int): Number of output categories.
    """
    def __init__(self):
        super(AudioNet, self).__init__()
        self.conv1 = qnn.QuantConv1d(1, 16, kernel_size=3, stride=1, padding=1, weight_bit_width=8)
        self.relu1 = qnn.QuantReLU(bit_width=8)
        self.pool1 = qnn.QuantAvgPool1d(kernel_size=2)

        self.conv2 = qnn.QuantConv1d(16, 32, kernel_size=3, stride=1, padding=1, weight_bit_width=8)
        self.relu2 = qnn.QuantReLU(bit_width=8)
        self.pool2 = qnn.QuantAvgPool1d(kernel_size=2)

        self.fc1 = qnn.QuantLinear(32 * 10, 64, bias=True, weight_bit_width=8)
        self.relu3 = qnn.QuantReLU(bit_width=8)
        self.fc2 = qnn.QuantLinear(64, 10, bias=True, weight_bit_width=8)  # 10 Klassen

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
