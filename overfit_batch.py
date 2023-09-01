import hydra
import torch
from torch import nn, optim
from omegaconf import DictConfig

from train import set_seed
from model.VQVAE import VQVAE
from utils.data_loader import create_loaders
from visualizations import show_loss, compare_values


@hydra.main(version_base=None, config_path="configs", config_name="config_single_batch")
def overfit_single_batch(cfg: DictConfig):
    # Initialize model, loss, optimizer
    model = VQVAE(cfg.model, cfg.system.device)
    model.to(cfg.system.device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    # Create data loaders
    train_loader, _, _ = create_loaders(cfg, seed=cfg.system.seed)

    # Fetch a single batch of data
    single_batch = next(iter(train_loader))
    x_combined = torch.stack([single_batch["start"], single_batch["duration"], single_batch["velocity"]], dim=1)
    x_combined = x_combined.to(cfg.system.device)
    losses_dict = {
        "reconstruction_loss": [],
        "vq_loss": [],
    }

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        optimizer.zero_grad()

        reconstructed_x, vq_loss, losses, perplexity, _, _ = model(x_combined)

        # Compute the loss
        reconstruction_loss = criterion(reconstructed_x, x_combined)
        loss = reconstruction_loss + vq_loss
        losses["reconstruction_loss"] = reconstruction_loss.item()
        losses["loss"] = loss.item()

        # Backward and optimize
        loss.backward()
        optimizer.step()

        losses_dict["reconstruction_loss"].append(reconstruction_loss.item())
        losses_dict["vq_loss"].append(vq_loss.item())

        print(
            f"Epoch [{epoch}/{cfg.train.epochs}]: Recon loss: {losses_dict['reconstruction_loss'][-1]}, VQ loss: {losses_dict['vq_loss'][-1]}"
        )

    # visualize the original and reconstructed data
    lr = cfg.train.lr
    compare_values(
        start=single_batch["start"][0, :].detach().cpu().numpy(),
        duration=single_batch["duration"][0, :].detach().cpu().numpy(),
        velocity=single_batch["velocity"][0, :].detach().cpu().numpy(),
        start_recon=reconstructed_x[0, 0, :].detach().cpu().numpy(),
        duration_recon=reconstructed_x[0, 1, :].detach().cpu().numpy(),
        velocity_recon=reconstructed_x[0, 2, :].detach().cpu().numpy(),
        title=f"Overfitting a single batch, lr={lr}",
        lr=lr,
    )

    # visualize the losses
    show_loss(losses_dict, lr, title=f"Overfitting a single batch, lr={lr}")


if __name__ == "__main__":
    set_seed(42)
    overfit_single_batch()
