import os

import torch
import numpy as np
import pretty_midi
from fortepyan.audio import render as render_audio

from model.VQVAE import VQVAE


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


def load_checkpoint(ckpt_path: str):
    checkpoint = torch.load(ckpt_path)
    cfg = checkpoint["config"]

    model = VQVAE(cfg.model, cfg.system.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint, cfg, model


def render_midi_to_mp3(filename: str, pitch: np.ndarray, dstart: np.ndarray, duration: np.ndarray, velocity: np.ndarray) -> str:
    midi_filename = os.path.basename(filename)
    mp3_path = os.path.join("tmp", midi_filename)

    if not os.path.exists(mp3_path):
        track = to_midi(pitch, dstart, duration, velocity)
        render_audio.midi_to_mp3(track, mp3_path)

    return mp3_path


def compare_original_generated(original: dict, generated: torch.Tensor):
    for track_idx in range(original["start"].shape[0]):
        original_pitch = original["pitch"][track_idx, :].detach().cpu().numpy()
        original_dstart = original["start"][track_idx, :].detach().cpu().numpy()
        original_duration = original["duration"][track_idx, :].detach().cpu().numpy()
        original_velocity = original["velocity"][track_idx, :].detach().cpu().numpy()

        generated_dstart = generated[track_idx, 0, :].detach().cpu().numpy()
        generated_duration = generated[track_idx, 1, :].detach().cpu().numpy()
        generated_velocity = generated[track_idx, 2, :].detach().cpu().numpy()

        # denormalize velocity
        original_velocity = original_velocity * 127.0
        generated_velocity = generated_velocity * 127.0
        generated_velocity = np.round(generated_velocity)

        original_mp3_path = render_midi_to_mp3(
            f"{track_idx}_original.mp3", original_pitch, original_dstart, original_duration, original_velocity
        )
        generated_mp3_path = render_midi_to_mp3(
            f"{track_idx}_reconstructed.mp3", original_pitch, generated_dstart, generated_duration, generated_velocity
        )

        print(f"Original: {original_mp3_path}")
        print(f"Generated: {generated_mp3_path}")


if __name__ == "__main__":
    ckpt_path = "checkpoints/2023_09_05_14_07_single_batch.pt"
    checkpoint, cfg, model = load_checkpoint(ckpt_path)

    single_batch = checkpoint["single_batch"]
    x_combined = torch.stack([single_batch["start"], single_batch["duration"], single_batch["velocity"]], dim=1)

    reconstructed_x, vq_loss, losses, perplexity, _, _ = model(x_combined)

    compare_original_generated(single_batch, reconstructed_x)
