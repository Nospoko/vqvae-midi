import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


def show_midi_random_pitch(start, duration, velocity, start_recon, duration_recon, velocity_recon, title):
    # reapeat 60, 62, 64, 65, 67 for pitch
    pitch = np.repeat(np.array([60, 62, 64, 65, 67]), start.shape[0] // 5)

    # de-normalize velocity from [-1, 1] to [0, 127]
    velocity = (velocity + 1) * 127.0 / 2.0
    velocity_recon = (velocity_recon + 1) * 127.0 / 2.0
    # if velocity > 127, set it to 127
    velocity_recon[velocity_recon > 127] = 127

    # take only first n notes
    n = 10
    start = start[:n]
    duration = duration[:n]
    velocity = velocity[:n]
    start_recon = start_recon[:n]
    duration_recon = duration_recon[:n]
    velocity_recon = velocity_recon[:n]
    pitch = pitch[:n]

    # 2 charts next to each other
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 5))
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


# TODO: Have to re-write this function after switch to ff.MidiPiece
def compare_values(start, duration, velocity, start_recon, duration_recon, velocity_recon, title, lr=None, num=0):
    """
    3 axes showing the parameters of the original and reconstructed data
    """
    # velocity = (velocity + 1) * 127.0 / 2.0
    # velocity_recon = (velocity_recon + 1) * 127.0 / 2.0

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 9))

    ax1.plot(start, color="blue", label="Original")
    ax1.plot(start_recon, color="orange", label="Reconstructed")
    ax1.set_title("Start")
    ax1.legend()

    ax2.plot(duration, color="blue", label="Original")
    ax2.plot(duration_recon, color="orange", label="Reconstructed")
    ax2.set_title("Duration")
    ax2.legend()

    ax3.plot(velocity, color="blue", label="Original")
    ax3.plot(velocity_recon, color="orange", label="Reconstructed")
    ax3.set_title("Velocity")
    ax3.legend()

    plt.suptitle(title)
    # save the figure
    plt.tight_layout()
    plt.savefig(f"tmp/graphs/{num}_values_{lr}.png")


def show_loss(losses: dict, lr, title="losses"):
    """
    Show the losses of the model
    """
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.set_title(title)

    color = "tab:red"
    ax.set_xlabel("Epoch")
    ax.set_ylabel("reconstruction loss", color=color)
    ax.plot(losses["reconstruction_loss"], color=color)
    ax.tick_params(axis="y", labelcolor=color)

    ax2 = ax.twinx()

    color = "tab:blue"
    ax2.set_ylabel("vq loss", color=color)
    ax2.plot(losses["vq_loss"], color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.savefig(f"results/losses{lr}.png")


def draw_ecg_reconstructions(
    model: nn.Module,
    signals: torch.Tensor,
) -> plt.Figure:
    reconstructed_x, vq_loss, losses, perplexity, _, _ = model(signals)
    reconstructions = reconstructed_x.detach()
    signals = signals.cpu()
    n_samples = signals.shape[0]

    fig, axes = plt.subplots(ncols=2, nrows=n_samples, figsize=[10, 2 * n_samples])
    for it in range(n_samples):
        left_ax = axes[it][0]
        right_ax = axes[it][1]
        left_ax.plot(signals[it][0], label="signal")
        left_ax.plot(reconstructions[it][0], label="reconstruction")
        left_ax.legend()

        right_ax.plot(signals[it][1], label="signal")
        right_ax.plot(reconstructions[it][1], label="reconstruction")
        right_ax.legend()

    return fig


@torch.no_grad()
def visualize_ecg_reconstruction(cfg, model, test_loader):
    model.eval()
    to_plot = []

    # Getting batches from the test_loader until we have enough signals
    for batch in test_loader:
        signals = batch["signal"]
        higher_than = 0.8
        heartbeat_signals = [signal for signal in signals if signal.max() > higher_than]
        to_plot.extend(heartbeat_signals)
        if len(to_plot) >= 6:  # Exit loop when we have at least 6 signals
            break

    # If there are not enough signals, give a message and exit
    if len(to_plot) < 4:
        print("Not enough signals with max value > 0.8 found.")
        return

    # Convert list to tensor and pass through autoencoder
    to_plot_tensor = torch.stack(to_plot[:4])
    reconstructed_x, vq_loss, losses, perplexity, _, _ = model(to_plot_tensor)

    reconstructions = reconstructed_x.detach()

    # Convert to CPU for visualization
    to_plot_tensor = to_plot_tensor.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()

    # Number of samples to visualize (can be less than 4 if not enough signals found)
    num_samples = min(4, len(to_plot))

    plt.figure(figsize=(20, 6 * num_samples))

    for i in range(num_samples):
        # Channel 0
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.plot(to_plot_tensor[i, 0, :], label="Original Channel 0", color="blue")
        plt.plot(reconstructions[i, 0, :], label="Reconstructed Channel 0", color="red", linestyle="--")

        # Channel 1
        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.plot(to_plot_tensor[i, 1, :], label="Original Channel 1", color="green")
        plt.plot(reconstructions[i, 1, :], label="Reconstructed Channel 1", color="orange", linestyle="--")

    plt.tight_layout(pad=5.0)
    # save the plot
    plt.savefig("{}/reconstructions_{}.png".format("tmp", "ecg"))
    plt.show()
