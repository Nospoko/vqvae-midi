import os

import torch
import numpy as np
import pretty_midi
from fortepyan.audio import render as render_audio

from utils.visualizations import compare_values


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


# Might want to rename this to something else
@torch.no_grad()
def generate_midi(cfg, model, batch, filename: str, track_idx: int = 0, midi: bool = True, mp3: bool = True):
    model.eval()

    x_combined = torch.stack([batch["start"], batch["duration"], batch["velocity"]], dim=1)
    # x_combined = x_combined.to(cfg.system.device)

    reconstructed_x, vq_loss, losses, perplexity, _, _ = model(x_combined)
    original_pitch = batch["pitch"][track_idx, :].detach().cpu().numpy()
    original_dstart = batch["start"][track_idx, :].detach().cpu().numpy()
    original_duration = batch["duration"][track_idx, :].detach().cpu().numpy()
    original_velocity = batch["velocity"][track_idx, :].detach().cpu().numpy()

    generated_dstart = reconstructed_x[track_idx, 0, :].detach().cpu().numpy()
    generated_duration = reconstructed_x[track_idx, 1, :].detach().cpu().numpy()
    generated_velocity = reconstructed_x[track_idx, 2, :].detach().cpu().numpy()

    # denormalize velocity
    original_velocity = original_velocity * 127.0
    generated_velocity = generated_velocity * 127.0
    generated_velocity = np.round(generated_velocity)

    if midi:
        save_midi(
            track=to_midi(
                pitch=original_pitch,
                dstart=original_dstart,
                duration=original_duration,
                velocity=original_velocity,
            ),
            filename=f"{track_idx}_original.mid",
        )
        save_midi(
            track=to_midi(
                pitch=original_pitch,
                dstart=generated_dstart,
                duration=generated_duration,
                velocity=generated_velocity,
            ),
            filename=f"{track_idx}_reconstructed.mid",
        )

    if mp3:
        render_midi_to_mp3(
            filename=f"original/{track_idx}_original.mp3",
            pitch=original_pitch,
            dstart=original_dstart,
            duration=original_duration,
            velocity=original_velocity,
            original=True,
        )
        render_midi_to_mp3(
            filename=f"reconstructed/{track_idx}_reconstructed.mp3",
            pitch=original_pitch,
            dstart=generated_dstart,
            duration=generated_duration,
            velocity=generated_velocity,
            original=False,
        )
        compare_values(
            start=original_dstart,
            duration=original_duration,
            velocity=original_velocity,
            start_recon=generated_dstart,
            duration_recon=generated_duration,
            velocity_recon=generated_velocity,
            title=f"{filename}_{track_idx}",
            lr=cfg.train.lr,
            num=track_idx,
        )

    return {
        "original": {
            "pitch": original_pitch,
            "dstart": original_dstart,
            "duration": original_duration,
            "velocity": original_velocity,
        },
        "generated": {
            "pitch": original_pitch,
            "dstart": generated_dstart,
            "duration": generated_duration,
            "velocity": generated_velocity,
        },
    }
