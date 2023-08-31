import hydra
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from omegaconf import OmegaConf, DictConfig

import wandb
from model.VQVAE import VQVAE
from utils.data_loader import create_loaders


def count_parameters(model):
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    print(f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")


def set_seed(seed):
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def test(model, test_loader, criterion, cfg, epoch):
    model.eval()
    total_test_loss = 0
    total_reconstruction_loss = 0
    n_test = 0

    with torch.no_grad():
        for batch in test_loader:
            x_combined = torch.stack([batch["start"], batch["duration"], batch["velocity"]], dim=1)

            if cfg.system.cuda:
                x_combined = x_combined.to("cuda")

            reconstructed_x, vq_loss, losses, perplexity, _, _ = model(x_combined)

            # Compute the loss
            reconstruction_loss = criterion(reconstructed_x, x_combined)
            total_reconstruction_loss += reconstruction_loss.item()

            loss = reconstruction_loss + vq_loss
            total_test_loss += loss.item()
            n_test += 1

        avg_test_loss = total_test_loss / n_test
        avg_reconstruction_loss = total_reconstruction_loss / n_test

        # Log the test metrics to wandb
        wandb.log(
            {
                "test/Average Test Loss": avg_test_loss,
                "test/Reconstruction Loss": avg_reconstruction_loss,
            }
        )
        print(f"Epoch {epoch}, Average Test Loss: {avg_test_loss}")


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

        x_combined = x_combined.to(cfg.system.device)

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

        if batch_idx + 1 % cfg.train.log_interval == 0:
            wandb.log(
                {
                    "train/Total Train Loss": total_train_loss / n_train,
                    "train/Reconstruction Loss": total_recon_error / n_train,
                    "train/VQ Loss": vq_loss.item(),
                    "train/Perplexity": perplexity,
                }
            )
        # Update tqdm loop
        progress_bar.set_description(f"Epoch {epoch} Loss: {losses['loss']:.4f}")

    # Save checkpoint
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, f"checkpoints/checkpoint_{epoch}.pt")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    name = f"MIDI_VQ-VAE_{cfg.run_date}"
    wandb.init(
        project="MIDI VQ-VAE",
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    model = VQVAE(cfg.model, cfg.system.device)
    model.to(cfg.system.device)
    # count_parameters(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    train_loader, _, _ = create_loaders(cfg)

    # Train the model
    for epoch in range(1, cfg.train.epochs + 1):
        train(model, train_loader, optimizer, criterion, cfg, epoch)
        test(model, train_loader, criterion, cfg, epoch)


if __name__ == "__main__":
    main()
