import torch
from huggingface_hub.file_download import hf_hub_download

from model.VQVAE import VQVAE


def load_checkpoint(ckpt_path: str = None):
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path)
    else:
        checkpoint = torch.load(hf_hub_download("SneakyInsect/VQ-VAE-MIDI", filename="2023_09_06_21_32_all_data165.pt"))
    cfg = checkpoint["config"]

    model = VQVAE(cfg.model, cfg.system.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint, cfg, model
