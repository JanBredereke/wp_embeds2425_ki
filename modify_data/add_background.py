import random

from pydub import AudioSegment


def add_background(sound: AudioSegment, background_noise: AudioSegment) -> AudioSegment:
    """
    @author <Friederike Korte>

    Adds given background noise to given sound. Randomized if background noise is longer than sound.
    Args:
        sound: AudioSegment to add background noise to
        background_noise: AudioSegment with background noise

    Returns: Sound (AudioSegment) with added background noise

    """
    if background_noise.duration_seconds > sound.duration_seconds:
        sound = add_random_part_of_background(sound=sound, background_noise=background_noise)

    else:
        sound = sound.overlay(background_noise, loop=True)

    return sound


def add_random_part_of_background(sound: AudioSegment, background_noise: AudioSegment) -> AudioSegment:
    """
    @author <Friederike Korte>

    Adds random part of given background noise to given sound.
    Args:
        sound: AudioSegment to add background noise to
        background_noise: AudioSegment with background noise

    Returns: Sound (AudioSegment) with added background noise

    """
    length_background_ms = background_noise.duration_seconds * 1000
    length_sound_ms = sound.duration_seconds * 1000
    start = random.uniform(0.0, length_background_ms - length_sound_ms)

    sound = sound.overlay(background_noise[start:])

    return sound
