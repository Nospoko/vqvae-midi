import numpy as np
from matplotlib import pyplot as plt

from utils.ecg_loader import create_ecg_loaders
from evals.checkpoint_tools import load_checkpoint
from utils.visualizations import draw_ecg_reconstructions

if __name__ == "__main__":
    ckpt_path = "checkpoints/2023_09_11_11_29_ECG_4.pt"
    checkpoint, cfg, model = load_checkpoint(ckpt_path)

    _, validation_loader, _ = create_ecg_loaders(cfg, seed=cfg.system.seed)

    model.eval()
    n_samples = 16
    idxs = np.random.randint(len(validation_loader.dataset), size=n_samples)
    signals = validation_loader.dataset[idxs]["signal"]
    draw_ecg_reconstructions(model, signals)
    savepath = "tmp/ecg-vqvae-reconstruction.png"
    plt.tight_layout()
    plt.savefig(savepath)
    print("Saved an image!", savepath)
