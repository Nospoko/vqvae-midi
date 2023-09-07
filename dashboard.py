import streamlit as st

from utils.data_loader import create_loaders
from evals.checkpoint_tools import load_checkpoint
from evals.midi_tools import generate_midi, plot_piano_roll, render_midi_to_mp3


def display_audio(title, fortepyan_midi, generated_fortepyan_midi, split, selected_batch, selected_track):
    st.title(title)

    # layout two columns for Original and Reconstructed
    col1, _, col2 = st.columns([2, 0.5, 2])

    with col1:
        st.write("### Original")
        fig = plot_piano_roll(fortepyan_midi)
        original_mp3_path = render_midi_to_mp3(
            piece=fortepyan_midi,
            filename=f"{split}_batch_{selected_batch}_track_{selected_track}_original.mp3",
            original=True,
        )
        st.pyplot(fig)
        st.audio(original_mp3_path, format="audio/mp3", start_time=0)

    with col2:
        st.write("### Reconstructed")
        fig2 = plot_piano_roll(generated_fortepyan_midi)
        generated_mp3_path = render_midi_to_mp3(
            piece=generated_fortepyan_midi,
            filename=f"{split}_batch_{selected_batch}_track_{selected_track}_reconstructed.mp3",
            original=False,
        )
        st.pyplot(fig2)
        st.audio(generated_mp3_path, format="audio/mp3", start_time=0)


def main():
    ckpt_path = "checkpoints/2023_09_06_21_32_all_data165.pt"
    _, cfg, model = load_checkpoint(ckpt_path)

    train_loader, val_loader, test_loader = create_loaders(cfg)

    with st.sidebar:
        st.header("Choose Data Split")
        split = st.selectbox("Split:", ["Train", "Validation", "Test"])
        selected_loader = train_loader if split == "Train" else (val_loader if split == "Validation" else test_loader)
        selected_batch = st.sidebar.selectbox("Choose a random batch:", list(range(len(selected_loader))))
        selected_track = st.sidebar.selectbox("Choose a random track:", list(range(cfg.train.batch_size)))

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

            display_audio(title, fortepyan_midi, generated_fortepyan_midi, split, selected_batch, selected_track)
            break


if __name__ == "__main__":
    main()
