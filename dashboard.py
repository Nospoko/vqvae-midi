import streamlit as st

from utils.data_loader import create_loaders
from evals.checkpoint_tools import load_checkpoint
from evals.midi_tools import generate_midi, plot_piano_roll, render_midi_to_mp3


def main():
    st.title("VQ-VAE for Music Generation")

    # Load the checkpoint
    ckpt_path = "checkpoints/2023_09_06_21_32_all_data165.pt"
    _, cfg, model = load_checkpoint(ckpt_path)

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
            title, fortepyan_midi, generated_fortepyan_midi = generate_midi(
                cfg,
                model,
                batch,
                f"{split}_batch_{selected_batch}",
                track_idx=selected_track,
                mp3=False,
                midi=False,
            )
            # write the title of the piece
            st.write(f"## {title}")

            # Original audio:
            st.write("### Original")
            fig = plot_piano_roll(fortepyan_midi)

            # render the original audio
            original_mp3_path = render_midi_to_mp3(
                piece=fortepyan_midi,
                filename=f"{split}_batch_{selected_batch}_track_{selected_track}_original.mp3",
                original=True,
            )
            # display piano roll and audio
            st.pyplot(fig)
            st.audio(original_mp3_path, format="audio/mp3", start_time=0)

            # Reconstructed audio:
            st.write("### Reconstructed")
            fig2 = plot_piano_roll(generated_fortepyan_midi)

            generated_mp3_path = render_midi_to_mp3(
                piece=generated_fortepyan_midi,
                filename=f"{split}_batch_{selected_batch}_track_{selected_track}_reconstructed.mp3",
                original=False,
            )

            st.pyplot(fig2)
            st.audio(generated_mp3_path, format="audio/mp3", start_time=0)

            break


if __name__ == "__main__":
    main()
