import os

import torch
import numpy as np
import pretty_midi
from fortepyan.audio import render as render_audio

from model.VQVAE import VQVAE
from visualizations import compare_values
from utils.data_loader import create_loaders


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


@torch.no_grad()
def generate_midi(cfg, model, batch, filename: str, track_idx: int = 0):
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

    render_midi_to_mp3(
        filename=f"{filename}_{track_idx}_original.mp3",
        pitch=original_pitch,
        dstart=original_dstart,
        duration=original_duration,
        velocity=original_velocity,
    )
    render_midi_to_mp3(
        filename=f"{filename}_{track_idx}_reconstructed.mp3",
        pitch=original_pitch,
        dstart=generated_dstart,
        duration=generated_duration,
        velocity=generated_velocity,
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


if __name__ == "__main__":
    ckpt_path = "checkpoints/2023_09_06_11_24_all_data17.pt"
    checkpoint, cfg, model = load_checkpoint(ckpt_path)

    _, validation_loader, _ = create_loaders(cfg, seed=cfg.system.seed)

    # take a batch from the validation set
    batch = next(iter(validation_loader))
    print(batch["pitch"].shape)

    generate_midi(
        cfg,
        model,
        batch,
        "all_data17",
        track_idx=0,
    )
