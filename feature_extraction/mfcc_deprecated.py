import numpy as np
from python_speech_features import mfcc


def mfcc_deprecated(raw_audios: np.ndarray) -> np.ndarray:
    """
    Extract features from given audios using MFCC. For further details, see https://python-speech-features.readthedocs.io/en/latest/
    :param raw_audios: audios tu use for feature extraction
    :return: np.array with features extracted
    """
    features = mfcc(
        signal=raw_audios
    )
    print(f'Features: {features.shape}')

    return features
