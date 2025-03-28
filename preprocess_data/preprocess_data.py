import numpy as np
from torch import Tensor

from enums.data_classes import AudioData
from preprocess_data.split_samples_in_train_and_test_data import split_samples_in_train_and_test_data


def preprocess_data(
        samples_dict: dict[str, list[AudioData]],
        with_background_noise: bool
) -> tuple[list[set[Tensor, int, str, np.array]], list[set[Tensor, int, str, np.array]], list, int]:
    """
    @author <Friederike Korte>

    Splits data into train and test sets.
    If with_background_noise is true, the samples wth background noise will be used, otherwise the samples without
    background noise will be used.
    Args:
        samples_dict: dictionary with all samples
                      (str = label/category, list[AudioData] = list of all AudioData-Objects for this label/category)
        with_background_noise: if samples with or without background noise shall be used

    Returns: list of train and test sets, list of labels, extracted mfcc-features

    """
    labels: list = list(samples_dict.keys())

    train_data, test_data = split_samples_in_train_and_test_data(
        samples_dict=samples_dict, with_background_noise=with_background_noise
    )

    in_features = samples_dict['back'][0].features.shape[0]

    print(f'\nTrain samples: {len(train_data)}\nTest samples: {len(test_data)}')

    return train_data, test_data, labels, in_features
