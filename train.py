import hydra
import torch
import wandb
from torch import nn, optim
from omegaconf import OmegaConf, DictConfig

from model.VQVAE import VQVAE
from utils.data_loader import create_loaders


def count_parameters(model):
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    print(f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")


def train(model, train_loader, optimizer, criterion, cfg, epoch):
    model.train()
    total_train_loss = 0
    total_recon_error = 0
    n_train = 0
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        # BTC -> BCT

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

        if (batch_idx + 1) % cfg.train.log_interval == 0:
            perplexity_value = perplexity.item()
            print("Epoch {}: loss {:.4f} perplexity {:.3f}".format(epoch, losses["loss"], perplexity_value))


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
