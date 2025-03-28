import logging
import os.path

import numpy as np
import pydub
import torch
from pydub import AudioSegment
from pydub.utils import mediainfo
from tqdm import tqdm

import enums.colors as colors
from enums.data_classes import AudioData
from feature_extraction.mfcc import mfcc
from modify_data.add_background import add_background
from modify_data.change_length_of_audio import change_length_of_audio


def read_audio_files_from_folder_into_list(
        folder: str,
        blank_audio: AudioSegment,
        background_noise: AudioSegment
) -> list[AudioData]:
    """
    @author <Friederike Korte>

    Reads all audio files from given folder and returns a list of AudioData objects.
    Args:
        folder: Folder containing audio files.
        blank_audio: Blank audio file, used for changing length of audio from folder.
        background_noise: Background noise, used for adding background noise to audios from folder.

    Returns: list of AudioData objects

    """
    file_list = os.listdir(folder)
    audio_files_list = []

    backslash_char = "\\"
    for file in tqdm(file_list, desc=f"\t... {folder.split(backslash_char)[-1]}"):
        file_path = os.path.join(folder, file)

        if os.path.isfile(file_path):
            try:
                audio_data = read_mp3_file_into_audio_data(
                    audio_file_path=file_path, blank_audio=blank_audio, background_noise=background_noise
                )
                if audio_data is not None:
                    audio_files_list.append(audio_data)

            except Exception as e:
                print(e)

        else:
            logging.getLogger(__name__).warning(
                f'{colors.YELLOW}File Path "{file_path}" is not a valid file path.'
                f'\nSkipping this file.\n{colors.COLOR_OFF}'
            )

    return audio_files_list


def read_mp3_file_into_audio_data(
        audio_file_path: str,
        blank_audio: AudioSegment,
        background_noise: AudioSegment
) -> AudioData | None:
    """
    @author <Friederike Korte>

    Reads audio file from given path, alternates it and returns it as an AudioData object.
    Args:
        audio_file_path: path to audio file
        blank_audio: Blank audio file, used for changing length of audio
        background_noise: Background noise, used for adding background noise to audio

    Returns: audio file as AudioData object

    """
    if os.path.isfile(audio_file_path):
        # get sample rate
        info = mediainfo(audio_file_path)
        sample_rate = int(info['sample_rate'])

        # Load the file
        sound = pydub.AudioSegment.from_mp3(audio_file_path)

        # feature extraction with mfcc
        raw_data = np.array(sound.get_array_of_samples())
        features = mfcc(raw_audio=raw_data, sample_rate=sample_rate)

        # change length of audio
        sound_without_bg_noise = change_length_of_audio(sound=sound, blank_audio=blank_audio)
        waveform_without_bg_noise = torch.from_numpy(np.array(sound_without_bg_noise.get_array_of_samples()))

        # add backgound noise
        sound_with_bg_noise = add_background(sound=sound_without_bg_noise, background_noise=background_noise)
        waveform_with_bg_noise = torch.from_numpy(np.array(sound_with_bg_noise.get_array_of_samples()))

        # Create audio data and return
        audio_data = AudioData(
            sample_rate=sample_rate,
            waveform_raw=torch.from_numpy(np.array(sound_with_bg_noise.get_array_of_samples())),
            waveform_without_bg_noise=waveform_without_bg_noise,
            waveform_with_bg_noise=waveform_with_bg_noise,
            file_name=audio_file_path,
            features=features
        )
        return audio_data

    return None
