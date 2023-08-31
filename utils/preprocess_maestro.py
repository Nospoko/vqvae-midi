import os

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset


def create_dict_from_split(split, rolling_window_size=1024, hop_size=256):
    # Initialize lists to hold data for each window from all tracks
    names = []
    starts = []
    durations = []
    ends = []
    pitches = []
    velocities = []

    for track in split:
        name = f"{track['composer']} {track['title']}"
        total_length = len(track["notes"]["start"])

        window_number = 0
        for idx in range(0, total_length - rolling_window_size + 1, hop_size):
            names.append(f"{window_number} {name}")

            start_window = np.array(track["notes"]["start"][idx : idx + rolling_window_size])
            first_start_value = start_window[0]
            start_window -= first_start_value
            starts.append(start_window.tolist())

            durations.append(track["notes"]["duration"][idx : idx + rolling_window_size])
            # Shift "end" accordingly
            end_window = np.array(track["notes"]["end"][idx : idx + rolling_window_size])
            end_window -= first_start_value
            ends.append(end_window.tolist())

            pitches.append(track["notes"]["pitch"][idx : idx + rolling_window_size])
            velocity = track["notes"]["velocity"][idx : idx + rolling_window_size]
            # Convert velocity to a numpy array and normalize to [-1, 1]
            velocity = np.array(velocity)
            velocity = 2 * ((velocity - 0) / (127 - 0)) - 1
            velocities.append(velocity.tolist())

            # change name for next window
            window_number += 1

    # Create a dictionary containing all the data
    data_dict = {
        "name": names,
        "start": starts,
        "duration": durations,
        "end": ends,
        "pitch": pitches,
        "velocity": velocities,
    }

    return data_dict


if __name__ == "__main__":
    data = load_dataset("roszcz/maestro-v1-sustain")
    rolling_window_size = 60
    hop_size = 20

    train = data["train"]
    validation = data["validation"]
    test = data["test"]

    train_dict = create_dict_from_split(train, rolling_window_size, hop_size)
    validation_dict = create_dict_from_split(validation, rolling_window_size, hop_size)
    test_dict = create_dict_from_split(test, rolling_window_size, hop_size)

    DATASET_NAME = "SneakyInsect/maestro-preprocessed"
    HUGGINGFACE_TOKEN = os.environ["HUGGINGFACE_TOKEN"]

    train_dataset = Dataset.from_dict(train_dict)
    validation_dataset = Dataset.from_dict(validation_dict)
    test_dataset = Dataset.from_dict(test_dict)

    dataset_dict = DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})

    dataset_dict.push_to_hub(DATASET_NAME, token=HUGGINGFACE_TOKEN)
