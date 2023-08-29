import hydra
import wandb
from omegaconf import OmegaConf, DictConfig

from utils.data_loader import create_loaders


def count_parameters(model):
    print(f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.logger.enable_wandb:
        # TODO: change accoridngly
        name = f"{cfg.dataset.name}_{cfg.run_date}"
        wandb.init(
            project="MIDI VQ-VAE",
            name=name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # model = VQVAE(cfg)
    train_loader, _, _ = create_loaders(cfg)
    # print one batch
    for batch in train_loader:
        print("min/max values for start, duration, velocity:")
        print(f"start: {batch['start'].min()}, {batch['start'].max()}")
        print(f"duration: {batch['duration'].min()}, {batch['duration'].max()}")
        print(f"velocity: {batch['velocity'].min()}, {batch['velocity'].max()}")
        break
    # count_parameters(model)


if __name__ == "__main__":
    main()
