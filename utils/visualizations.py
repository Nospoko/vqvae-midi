import numpy as np
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
