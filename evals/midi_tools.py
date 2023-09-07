import os

import numpy as np
import pretty_midi
from fortepyan.audio import render as render_audio


def to_midi(pitch: np.ndarray, dstart: np.ndarray, duration: np.ndarray, velocity: np.ndarray, track_name: str = "piano"):
    track = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name=track_name)

    previous_start = 0.0

    for p, ds, d, v in zip(pitch, dstart, duration, velocity):
        start = previous_start + ds
        end = start + d
        previous_start = start

        note = pretty_midi.Note(
            velocity=int(v),
            pitch=int(p),
            start=start,
            end=end,
        )

        piano.notes.append(note)

    track.instruments.append(piano)

    return track


def render_midi_to_mp3(
    filename: str, pitch: np.ndarray, dstart: np.ndarray, duration: np.ndarray, velocity: np.ndarray, original: bool
) -> str:
    midi_filename = os.path.basename(filename)

    if original:
        mp3_path = os.path.join("tmp", "original", midi_filename)
    else:
        mp3_path = os.path.join("tmp", "reconstructed", midi_filename)

    if not os.path.exists(mp3_path):
        track = to_midi(pitch, dstart, duration, velocity)
        render_audio.midi_to_mp3(track, mp3_path)

    return mp3_path


def save_midi(track: pretty_midi.PrettyMIDI, filename: str):
    # add tmp/midi directory to filename
    filename = os.path.join("tmp", "midi", filename)
    track.write(filename)
