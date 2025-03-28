import random

import numpy as np
from torch import Tensor

import state
from enums.data_classes import AudioData


def split_samples_in_train_and_test_data(
        samples_dict: dict[str, list[AudioData]],
        with_background_noise: bool
) -> tuple[list[set[Tensor, int, str, np.array]], list[set[Tensor, int, str, np.array]]]:
    train_data = []
    test_data = []

    for label, samples_list in samples_dict.items():
        # split random samples into train data
        amount_of_train_samples = int(len(samples_list) * state.config["training"]["train_data_ratio"])
        train_data_per_category = random.sample(samples_list, amount_of_train_samples)

        # add all samples, that have not been split into train data, to test data
        for sample in samples_list:
            sample: AudioData
            added_to_train_data = False

            waveform = sample.waveform_with_bg_noise if with_background_noise else sample.waveform_without_bg_noise

            for train_sample in train_data_per_category:
                if sample.file_name == train_sample.file_name:
                    train_data.append({waveform, sample.sample_rate, label})
                    added_to_train_data = True
                    break

            if not added_to_train_data:
                test_data.append({waveform, sample.sample_rate, label})

    return train_data, test_data
