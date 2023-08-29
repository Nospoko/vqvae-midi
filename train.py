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


def train(model, train_loader, optimizer, criterion, cfg, epoch, best_train_loss):
    model.train()
    total_train_loss = 0
    total_recon_error = 0
    n_train = 0
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        x_combined = torch.stack([batch["start"], batch["duration"], batch["velocity"]], dim=1)
        if cfg.system.cuda:
            x_combined = x_combined.to("cuda")

        output = model(x_combined)
        # the output returns:
        # return {
        #     "dictionary_loss": dictionary_loss,
        #     "commitment_loss": commitment_loss,
        #     "x_recon": x_recon,
        # }

        # Compute the loss
        recon_error = criterion(output["x_recon"], x_combined)
        total_recon_error += recon_error.item()

        loss = recon_error + cfg.vqvae.beta * output["commitment_loss"]

        if not cfg.vqvae.use_ema:
            loss += output["dictionary_loss"]

        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_train += 1

        if (batch_idx + 1) % cfg.train.log_interval == 0:
            avg_train_loss = total_train_loss / n_train
            avg_recon_error = total_recon_error / n_train

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(x_combined)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]"
            )
            print(f"Avg Train Loss: {avg_train_loss}")
            print(f"Best Train Loss: {best_train_loss}")
            print(f"Avg Recon Error: {avg_recon_error}")

            total_train_loss = 0
            total_recon_error = 0
            n_train = 0

    return best_train_loss


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.logger.enable_wandb:
        name = f"{cfg.dataset.name}_{cfg.run_date}"
        wandb.init(
            project="MIDI VQ-VAE",
            name=name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # TODO: Initialize your model here
    model = VQVAE(cfg)
    count_parameters(model)

    # TODO: Choose a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    train_loader, _, _ = create_loaders(cfg)

    for batch_idx, batch in enumerate(train_loader):
        print(type(batch["start"]), batch["start"].shape)
        print(type(batch["duration"]), batch["duration"].shape)
        print(type(batch["velocity"]), batch["velocity"].shape)
        break

    # Train the model
    best_train_loss = float("inf")
    for epoch in range(1, cfg.train.epochs + 1):
        best_train_loss = train(model, train_loader, optimizer, criterion, cfg, epoch, best_train_loss)


if __name__ == "__main__":
    main()
