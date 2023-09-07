import streamlit as st

from utils.data_loader import create_loaders
from evals.checkpoint_tools import load_checkpoint
from evals.midi_tools import generate_midi, render_midi_to_mp3  # Replace with the actual import


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
    selected_batch = st.selectbox("Choose a random batch:", batch_options)

    for i, batch in enumerate(selected_loader):
        if i == selected_batch:
            results = generate_midi(
                cfg,
                model,
                batch,
                f"{split}_batch_{selected_batch}",
                track_idx=0,
                mp3=False,
                midi=False,
            )

            # Generate and play the original and reconstructed audio
            original_mp3_path = render_midi_to_mp3(
                filename=f"original_{split}_batch_{selected_batch}.mp3",
                pitch=results["original"]["pitch"],
                dstart=results["original"]["dstart"],
                duration=results["original"]["duration"],
                velocity=results["original"]["velocity"],
                original=True,
            )

            generated_mp3_path = render_midi_to_mp3(
                filename=f"generated_{split}_batch_{selected_batch}.mp3",
                pitch=results["generated"]["pitch"],
                dstart=results["generated"]["dstart"],
                duration=results["generated"]["duration"],
                velocity=results["generated"]["velocity"],
                original=False,
            )

            st.audio(original_mp3_path, format="audio/mp3", start_time=0)
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
