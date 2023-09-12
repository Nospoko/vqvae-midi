import os

import hydra
import torch
import numpy as np
from omegaconf import DictConfig

from utils.data_loader import create_loaders
from evals.checkpoint_tools import load_checkpoint
from evals.midi_tools import save_midi, to_fortepyan_midi


@hydra.main(config_path="../configs", config_name="evaluation", version_base="1.3.2")
def process_folder(cfg: DictConfig) -> None:
    checkpoint, model_config, model = load_checkpoint()

    _, validation_loader, _ = create_loaders(model_config)

    model.eval()

    # create target folder
    os.makedirs(cfg.target_folder, exist_ok=True)

    # load the correct number of batches
    batches = []
    remaining_midi_files = cfg.num_midi_files
    batch_size = model_config.train.batch_size

    for i, batch in enumerate(validation_loader):
        batches.append(batch)
        remaining_midi_files -= batch_size  # Deduct the batch size from the remaining number of files
        if remaining_midi_files <= 0:
            break

    # save how_many midi files from each batch
    for i, batch in enumerate(batches):
        x_combined = torch.stack([batch["start"], batch["duration"], batch["velocity"]], dim=1)

        reconstructed_x, _, _, _, _, _ = model(x_combined)

        if i == len(batches) - 1:
            # Last batch, save only the remaining number of files
            how_many = remaining_midi_files + batch_size
        else:
            # Not the last batch, save the entire batch
            how_many = batch_size

        for track_idx in range(how_many):
            generated_velocity = reconstructed_x[track_idx, 2, :].detach().cpu().numpy()
            generated_velocity = generated_velocity * 127.0
            generated_velocity = np.round(generated_velocity)

            original_piece = to_fortepyan_midi(
                pitch=batch["pitch"][track_idx, :].detach().cpu().numpy(),
                dstart=batch["start"][track_idx, :].detach().cpu().numpy(),
                duration=batch["duration"][track_idx, :].detach().cpu().numpy(),
                velocity=batch["velocity"][track_idx, :].detach().cpu().numpy() * 127.0,
            )

            reconstructed_piece = to_fortepyan_midi(
                pitch=batch["pitch"][track_idx, :].detach().cpu().numpy(),
                dstart=reconstructed_x[track_idx, 0, :].detach().cpu().numpy(),
                duration=reconstructed_x[track_idx, 1, :].detach().cpu().numpy(),
                velocity=np.round(generated_velocity),
            )

            save_midi(
                track=original_piece,
                filename=f"batch{i}_{track_idx}_original.mid",
                folder=cfg.target_folder,
            )

            save_midi(
                track=reconstructed_piece,
                filename=f"batch{i}_{track_idx}_reconstructed.mid",
                folder=cfg.target_folder,
            )
    print(f"Saved 2x{cfg.num_midi_files} midi files to {cfg.target_folder}")


if __name__ == "__main__":
    process_folder()
