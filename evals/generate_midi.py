import torch
import numpy as np

from utils.data_loader import create_loaders
from evals.checkpoint_tools import load_checkpoint
from evals.midi_tools import generate_midi, render_midi_to_mp3


# This can go, leaving it for now in case we need it later
def single_batch_compare(original: dict, generated: torch.Tensor):
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
    ckpt_path = "checkpoints/2023_09_06_21_32_all_data165.pt"
    checkpoint, cfg, model = load_checkpoint(ckpt_path)

    # If we want to evaluate a single batch:
    single_batch_eval = False
    if single_batch_eval:
        single_batch = checkpoint["single_batch"]
        x_combined = torch.stack([single_batch["start"], single_batch["duration"], single_batch["velocity"]], dim=1)

        reconstructed_x, vq_loss, losses, perplexity, _, _ = model(x_combined)

        single_batch_compare(single_batch, reconstructed_x)
        exit()

    # Model evaluation
    # get the validation loader
    _, validation_loader, _ = create_loaders(cfg, seed=cfg.system.seed)
    # take a batch from the validation set
    batch = next(iter(validation_loader))
    title, original_piece, reconstructed_piece = generate_midi(
        cfg,
        model,
        batch,
        "all_data17",
        track_idx=0,
        mp3=False,
        midi=True,
    )
