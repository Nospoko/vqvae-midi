import torch

from model.VQVAE import VQVAE


def load_checkpoint(ckpt_path: str):
    checkpoint = torch.load(ckpt_path)
    cfg = checkpoint["config"]

    model = VQVAE(cfg.model, cfg.system.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint, cfg, model
