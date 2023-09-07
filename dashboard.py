import streamlit as st
import matplotlib.pyplot as plt

from utils.data_loader import create_loaders
from evals.checkpoint_tools import load_checkpoint
from evals.midi_tools import to_midi, generate_midi, render_midi_to_mp3


def plot_piano_roll(track, fs=100):
    piano_roll = track.get_piano_roll(fs=fs)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(piano_roll, aspect="auto", cmap="plasma")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Time (frames)")
    plt.ylabel("Pitch (MIDI)")

    return fig


def main():
    st.title("VQ-VAE for Music Generation")

    # Load the checkpoint
    ckpt_path = "checkpoints/2023_09_06_21_32_all_data165.pt"
    checkpoint, cfg, model = load_checkpoint(ckpt_path)

    # Get the loaders
    train_loader, val_loader, test_loader = create_loaders(cfg)

    # Sidebar for selecting data split
    with st.sidebar:
        st.header("Choose Data Split")
        split = st.selectbox("Split:", ["Train", "Validation", "Test"])

    # Use the appropriate loader based on user choice
    if split == "Train":
        selected_loader = train_loader
    elif split == "Validation":
        selected_loader = val_loader
    else:
        selected_loader = test_loader

    batch_options = list(range(len(selected_loader)))
    selected_batch = st.sidebar.selectbox("Choose a random batch:", batch_options)
    track_options = list(range(cfg.train.batch_size))
    selected_track = st.sidebar.selectbox("Choose a random track:", track_options)

    for i, batch in enumerate(selected_loader):
        if i == selected_batch:
            results = generate_midi(
                cfg,
                model,
                batch,
                f"{split}_batch_{selected_batch}",
                track_idx=selected_track,
                mp3=False,
                midi=False,
            )
            # write the title of the piece
            st.write(f"## {results['title']}")

            # Original audio:
            st.write("### Original")
            original_mp3_path = render_midi_to_mp3(
                filename=f"{selected_track}_original_{split}_batch_{selected_batch}.mp3",
                pitch=results["original"]["pitch"],
                dstart=results["original"]["dstart"],
                duration=results["original"]["duration"],
                velocity=results["original"]["velocity"],
                original=True,
            )
            original_track = to_midi(
                pitch=results["original"]["pitch"],
                dstart=results["original"]["dstart"],
                duration=results["original"]["duration"],
                velocity=results["original"]["velocity"],
            )
            fig = plot_piano_roll(original_track)
            st.pyplot(fig)
            st.audio(original_mp3_path, format="audio/mp3", start_time=0)

            # Reconstructed audio:
            st.write("### Reconstructed")
            generated_mp3_path = render_midi_to_mp3(
                filename=f"{selected_track}_generated_{split}_batch_{selected_batch}.mp3",
                pitch=results["generated"]["pitch"],
                dstart=results["generated"]["dstart"],
                duration=results["generated"]["duration"],
                velocity=results["generated"]["velocity"],
                original=False,
            )
            generated_track = to_midi(
                pitch=results["generated"]["pitch"],
                dstart=results["generated"]["dstart"],
                duration=results["generated"]["duration"],
                velocity=results["generated"]["velocity"],
            )
            fig2 = plot_piano_roll(generated_track)
            st.pyplot(fig2)
            st.audio(generated_mp3_path, format="audio/mp3", start_time=0)

            # Information table (replace with real metrics)
            # st.write("### Information Table")
            # st.table({
            #     "Metric": ["VQ Loss", "Other Losses", "Perplexity"],
            #     "Value": ["Value1", "Value2", "Value3"]
            # })

            break


if __name__ == "__main__":
    main()
