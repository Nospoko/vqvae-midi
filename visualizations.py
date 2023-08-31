import numpy as np
import matplotlib.pyplot as plt


def show_midi_random_pitch(start, duration, velocity, start_recon, duration_recon, velocity_recon, title):
    # reapeat 60, 62, 64, 65, 67 for pitch
    pitch = np.repeat(np.array([60, 62, 64, 65, 67]), 1024 // 5)

    # print value ranges:
    print("Reconstructed ranges: ")
    print(f"start: {np.min(start_recon)} - {np.max(start_recon)}")
    print(f"duration: {np.min(duration_recon)} - {np.max(duration_recon)}")
    print(f"velocity: {np.min(velocity_recon)} - {np.max(velocity_recon)}")

    print("Original ranges: ")
    print(f"start: {np.min(start)} - {np.max(start)}")
    print(f"duration: {np.min(duration)} - {np.max(duration)}")
    print(f"velocity: {np.min(velocity)} - {np.max(velocity)}")

    # 2 charts next to each other
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5))
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
