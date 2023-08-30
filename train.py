import hydra
import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch import nn, optim
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig

from model.VQVAE import VQVAE
from utils.data_loader import create_loaders


def count_parameters(model):
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    print(f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")


def show_midi_random_pitch(start, duration, velocity, start_recon, duration_recon, velocity_recon, title):
    # reapeat 60, 62, 64, 65, 67 for pitch
    pitch = np.repeat(np.array([60, 62, 64, 65, 67]), 1024 // 5)

    # print value ranges:
    print("Reconstructed ranges: ")
    print(f"start: {np.min(start_recon)} - {np.max(start_recon)}")
    print(f"duration: {np.min(duration_recon)} - {np.max(duration_recon)}")
    print(f"velocity: {np.min(velocity_recon)} - {np.max(velocity_recon)}")

    print("Original ranges: ")
    print(f"start: {np.min(start)} - {np.max(start)}")
    print(f"duration: {np.min(duration)} - {np.max(duration)}")
    print(f"velocity: {np.min(velocity)} - {np.max(velocity)}")

    # 2 charts next to each other
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5))
    for s, e, v, p in zip(start, duration, velocity, pitch):
        ax1.broken_barh([(s, e - s)], (p, 1), facecolors=(0.0, v / 127.0, 0.0))
        ax1.text(s + 0.5 * (e - s), p + 0.5, str(v), horizontalalignment="center", verticalalignment="center")

    for s, e, v, p in zip(start_recon, duration_recon, velocity_recon, pitch):
        ax2.broken_barh([(s, e - s)], (p, 1), facecolors=(0.0, v / 127.0, 0.0))
        ax2.text(s + 0.5 * (e - s), p + 0.5, str(v), horizontalalignment="center", verticalalignment="center")

    ax1.grid(True)
    ax1.set_title("Original")
    ax2.grid(True)
    ax2.set_title("Reconstructed")

    plt.suptitle(title)
    # save the figure
    plt.savefig(f"results/{title}.png")


def train(model, train_loader, optimizer, criterion, cfg, epoch):
    model.train()
    total_train_loss = 0
    total_recon_error = 0
    n_train = 0

    # Initialize tqdm loop
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

    for batch_idx, batch in progress_bar:
        optimizer.zero_grad()
        x_combined = torch.stack([batch["start"], batch["duration"], batch["velocity"]], dim=1)

        if cfg.system.cuda:
            x_combined = x_combined.to("cuda")

        reconstructed_x, vq_loss, losses, perplexity, _, _ = model(x_combined)

        # Compute the loss
        reconstruction_loss = criterion(reconstructed_x, x_combined)
        total_recon_error += reconstruction_loss.item()

        loss = reconstruction_loss + vq_loss
        losses["reconstruction_loss"] = reconstruction_loss.item()
        losses["loss"] = loss.item()

        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_train += 1

        # Update tqdm loop
        progress_bar.set_description(f"Epoch {epoch} Loss: {losses['loss']:.4f}")

        # if (batch_idx + 2) % cfg.train.log_interval == 0:
        #     perplexity_value = perplexity.item()
        #     print(f"Epoch {epoch}: loss {losses['loss']:.4f} perplexity {perplexity_value:.3f}")
        #     velocity = x_combined[0, 2].detach().cpu().numpy()
        #     velocity_recon = reconstructed_x[0, 2].detach().cpu().numpy()
        #     print(f"Velocity: {np.min(velocity)} - {np.max(velocity)}, shape: {velocity.shape}")
        #     print(f"Velocity recon: {np.min(velocity_recon)} - {np.max(velocity_recon)}, shape: {velocity_recon.shape}")

    # Save checkpoint
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, f"checkpoints/checkpoint_{epoch}.pt")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.logger.enable_wandb:
        name = f"{cfg.dataset.name}_{cfg.run_date}"
        wandb.init(
            project="MIDI VQ-VAE",
            name=name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    device = "cuda" if cfg.system.cuda else "cpu"

    model = VQVAE(cfg.model, device)
    model.to(device)
    count_parameters(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    train_loader, _, _ = create_loaders(cfg)

    # Train the model
    for epoch in range(1, cfg.train.epochs + 1):
        train(model, train_loader, optimizer, criterion, cfg, epoch)


if __name__ == "__main__":
    main()
