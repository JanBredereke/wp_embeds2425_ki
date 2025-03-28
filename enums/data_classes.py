from dataclasses import dataclass

import numpy as np
import torch


@dataclass()
class AudioData:
    waveform_raw: torch.Tensor
    waveform_without_bg_noise: torch.Tensor
    waveform_with_bg_noise: torch.Tensor
    sample_rate: int
    file_name: str
    features: np.ndarray


@dataclass()
class AudioDataSet:
    command_audios_list: list[AudioData]
    negative_audios_list: list[AudioData]
