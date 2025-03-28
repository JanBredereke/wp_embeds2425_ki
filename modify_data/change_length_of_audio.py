import random

from pydub import AudioSegment

import state


def change_length_of_audio(sound: AudioSegment, blank_audio: AudioSegment) -> AudioSegment:
    """
    @author <Friederike Korte>

    Changes length of sound with help of given blank_audio.
    Args:
        sound: sound (AudioSegment) to change length of
        blank_audio: AudioSegment to use for changing lenght of sound

    Returns: sound (AudioSegment) with changed length

    """
    wanted_length_ms = state.config["audio_length"]
    current_length_ms = sound.duration_seconds * 1000
    missing_length_ms = wanted_length_ms - current_length_ms

    split = random.uniform(0.0, missing_length_ms)

    first_part = blank_audio[:split]
    last_part = blank_audio[:missing_length_ms - split]

    sound = first_part + sound + last_part

    return sound
