import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


def create_ecg_loaders(cfg, seed=None):
    data = load_dataset("roszcz/ecg-segmentation-ltafdb")
    train = data["train"]
    validation = data["validation"]
    test = data["test"]

    columns = ["signal"]

    train.set_format(type="torch", columns=columns)
    validation.set_format(type="torch", columns=columns)
    test.set_format(type="torch", columns=columns)

    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    train_loader = DataLoader(train, batch_size=cfg.train.batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation, batch_size=cfg.train.batch_size, shuffle=False, generator=generator)
    test_loader = DataLoader(test, batch_size=cfg.train.batch_size, shuffle=False, generator=generator)

    return train_loader, validation_loader, test_loader
