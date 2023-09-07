import os

import torch
import numpy as np
import pandas as pd
import fortepyan as ff
from matplotlib import pyplot as plt
from fortepyan.audio import render as render_audio

from utils.visualizations import compare_values


def plot_piano_roll(piece: ff.MidiPiece, title: str = "Piano Roll") -> plt.Figure:
    fig = ff.view.draw_pianoroll_with_velocities(piece, title=title)
    return fig


def to_fortepyan_midi(
    pitch: np.ndarray,
    dstart: np.ndarray,
    duration: np.ndarray,
    velocity: np.ndarray,
) -> ff.MidiPiece:
    # change dstart to start, create end
    start = []
    start.append(dstart[0])
    for i in range(1, len(dstart)):
        start.append(start[i - 1] + dstart[i])

    end = []
    for i in range(len(start)):
        end.append(start[i] + duration[i])

    # pandas dataframe with pitch, start, end, velocity
    df = pd.DataFrame({"pitch": pitch, "start": start, "duration": duration, "end": end, "velocity": velocity})

    piece = ff.MidiPiece(df=df)

    return piece


def render_midi_to_mp3(piece: ff.MidiPiece, filename: str, original: bool = True) -> str:
    midi_filename = os.path.basename(filename)

    if original:
        mp3_path = os.path.join("tmp", "original", midi_filename)
    else:
        mp3_path = os.path.join("tmp", "reconstructed", midi_filename)

    if not os.path.exists(mp3_path):
        track = piece.to_midi()
        render_audio.midi_to_mp3(track, mp3_path)

    return mp3_path


def save_midi(track: ff.MidiPiece, filename: str):
    # add tmp/midi directory to filename
    filename = os.path.join("tmp", "midi", filename)
    track = track.to_midi()
    track.write(filename)


# Might want to rename this to something else
@torch.no_grad()
def generate_midi(
    cfg,
    model,
    batch,
    filename: str,
    track_idx: int = 0,
    midi: bool = True,
    mp3: bool = True,
) -> tuple[str, ff.MidiPiece, ff.MidiPiece]:
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

    original_piece = to_fortepyan_midi(
        pitch=original_pitch,
        dstart=original_dstart,
        duration=original_duration,
        velocity=original_velocity,
    )

    reconstructed_piece = to_fortepyan_midi(
        pitch=original_pitch,
        dstart=generated_dstart,
        duration=generated_duration,
        velocity=generated_velocity,
    )

    if midi:
        save_midi(
            track=original_piece,
            filename=f"{track_idx}_original.mid",
        )
        save_midi(
            track=reconstructed_piece,
            filename=f"{track_idx}_reconstructed.mid",
        )

    if mp3:
        render_midi_to_mp3(
            piece=original_piece,
            filename=f"original/{track_idx}_original.mp3",
            original=True,
        )
        render_midi_to_mp3(
            piece=reconstructed_piece,
            filename=f"reconstructed/{track_idx}_reconstructed.mp3",
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

    return batch["name"][track_idx], original_piece, reconstructed_piece
