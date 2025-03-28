import brevitas.nn as qnn
import torch.nn as nn

from models.BaseModelSequential import BaseModelSequential

class PreviousModel(BaseModelSequential):
    """
    @author <Friederike Korte>

    A sequential example model using SuperModelSeq with nn.Sequential.
    It's a Model from the previous group

    Args:
        in_features (int): Number of input features.
        number_of_categories (int): Number of output categories.
    """
    def __init__(self, in_features: int, number_of_categories: int):
        super().__init__()
        self.model = nn.Sequential(
            qnn.QuantLinear(in_features, in_features * 2, bias=True, weight_bit_width=3),
            nn.BatchNorm1d(in_features * 2),
            nn.Dropout(0.5),
            qnn.QuantReLU(bit_width=4),
            qnn.QuantLinear(in_features * 2, in_features * 2, bias=True, weight_bit_width=3),
            nn.BatchNorm1d(in_features * 2),
            nn.Dropout(0.5),
            qnn.QuantReLU(bit_width=4),
            qnn.QuantLinear(in_features * 2, in_features * 2, bias=True, weight_bit_width=3),
            nn.BatchNorm1d(in_features * 2),
            nn.Dropout(0.5),
            qnn.QuantReLU(bit_width=4),
            qnn.QuantLinear(in_features * 2, number_of_categories, bias=True, weight_bit_width=3)
        )
