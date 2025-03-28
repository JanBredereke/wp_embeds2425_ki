import logging
import os

from pydub import AudioSegment

import enums.colors as colors
import state
from enums.data_classes import AudioData
from load_data.read_audio_files_from_folder_into_array import read_audio_files_from_folder_into_list


def load_audio_files_from_folder() -> dict[str, list[AudioData]]:
    """
    @author <Friederike Korte>

    loads audio-files from folder data
    Returns: dictionary with audio-files as AudioData-Objects
            (str = label/category, list[AudioData] = list of all AudioData-Objects for this label/category)

    """
    print(f'{colors.CYAN}Loading audio files from folder...{colors.COLOR_OFF}')

    """load blank audio file"""
    blank_audio: AudioSegment = load_single_audio_file(audio_path=state.blank_audios_folder_path)
    if not blank_audio:
        logging.getLogger(__name__).warning(
            f'{colors.YELLOW}Did not find a blank audio. Skipping this step.{colors.COLOR_OFF}'
        )

    """load background audio file"""
    background_audio: AudioSegment = load_single_audio_file(audio_path=state.background_folder_path)
    if not background_audio:
        logging.getLogger(__name__).warning(
            f'{colors.YELLOW}Did not find a background audio. Skipping this step.{colors.COLOR_OFF}'
        )

    """load command and negative audio files into dict"""
    samples_dict = load_audios_into_samples_dict(
        blank_audio=blank_audio, background_audio=background_audio
    )

    print(f'{colors.CYAN}\nFinished loading audio files{colors.COLOR_OFF}')

    return samples_dict


def load_single_audio_file(audio_path: str) -> AudioSegment | None:
    """
    @author <Friederike Korte>

    Loads one mp3 audio file as an AudioSegment.
    Args:
        audio_path: path of audio file to load

    Returns: mps-sample as AudioSegment

    """
    if os.path.isdir(audio_path):
        audios = os.listdir(audio_path)
        for audio in audios:
            if '.mp3' in audio:
                return AudioSegment.from_file(os.path.join(audio_path, audio))

    return None


def load_audios_into_samples_dict(
        blank_audio: AudioSegment,
        background_audio: AudioSegment
) -> dict[str, list[AudioData]]:
    """
    @author <Friederike Korte>

    loads audio-files from subfolder in folder data into dict (label, list of AudioData)
    Args:
        blank_audio: blank audio file
        background_audio: background audio file

    Returns: dictionary with audio-files as AudioData-Objects

    """
    commands_subfolders = os.listdir(state.commands_folder_path)
    samples_dict = {}

    for subfolder in commands_subfolders:
        category = subfolder
        subfolder_path = os.path.join(state.commands_folder_path, subfolder)

        if os.path.isdir(subfolder_path):
            samples_list = read_audio_files_from_folder_into_list(
                folder=subfolder_path,
                blank_audio=blank_audio,
                background_noise=background_audio
            )
            samples_dict[category] = samples_list

    if os.path.isdir(state.negatives_folder_path):
        category = os.path.basename(state.negatives_folder_path)
        samples_list_negatives = read_audio_files_from_folder_into_list(
            folder=state.negatives_folder_path,
            blank_audio=blank_audio,
            background_noise=background_audio
        )
        samples_dict[category] = samples_list_negatives

    return samples_dict
