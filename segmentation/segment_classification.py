import os
import csv
import json
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import subprocess
from tqdm import tqdm


# Load YAMNet model and class names
def load_yamnet_model():
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = tf.keras.utils.get_file(
        'yamnet_class_map.csv',
        'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    )
    class_names = []
    with open(class_map_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            class_names.append(row[2])  # Third column has human-readable class names
    return model, class_names

def split_audio_to_segments(input_file, segment_duration, output_folder):
    """
    Split the input audio into fixed-length segments using ffmpeg.

    Args:
        input_file (str): Path to the input audio file.
        segment_duration (int): Duration of each segment in seconds.
        output_folder (str): Folder to store temporary audio segments.

    Returns:
        list: Paths to the generated segment files.
    """
    os.makedirs(output_folder, exist_ok=True)
    segment_pattern = os.path.join(output_folder, "segment%05d.wav")

    command = [
        "ffmpeg",
        "-i", input_file,
        "-f", "segment",
        "-segment_time", str(segment_duration),
        "-ar", "16000",  # Resample to 16kHz
        "-ac", "1",      # Convert to mono
        segment_pattern
    ]

    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # Collect all generated segment files
        return [os.path.join(output_folder, f) for f in sorted(os.listdir(output_folder)) if f.startswith("segment")]
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg error: {e.stderr.decode()}")

def classify_audio_segment(waveform, sr, yamnet_model):
    """
    Classify a single audio segment using the YamNet model.

    Args:
        waveform (np.ndarray): Audio waveform.
        sr (int): Sample rate of the waveform.
        yamnet_model: Loaded YamNet model.

    Returns:
        tuple: Raw prediction scores and predicted class indices.
    """
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)  # Convert to mono if not already
    if sr != 16000:
        raise ValueError(f"Sample rate must be 16 kHz, but got {sr} Hz.")

    waveform = waveform.astype(np.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    predicted_class_indices = tf.argmax(scores, axis=1).numpy()
    return scores.numpy(), predicted_class_indices

def process_audio(
        folder_path,
        yamnet_model,
        class_names,
        output_folder,
        top_n=3,
        segment_duration=2  # seconds per segment
):
    """
    Process audio files in a folder:
      1. Split audio into segments and save them as WAV files.
      2. Process each segment and classify it using the YamNet model.
      3. Save results to a CSV file.
      4. Clean up temporary segment files.

    Args:
        folder_path (str): Path to folder containing audio files.
        yamnet_model: Loaded YamNet TensorFlow model.
        class_names (list): Class names for YamNet.
        output_folder (str): Folder to save classification results.
        top_n (int): Number of top classes to save per segment.
        segment_duration (int): Duration of each audio segment in seconds.
    """
    temp_folder = "temp_segments"

    for file_name in tqdm(os.listdir(folder_path), desc="Processing Files"):
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing {file_path}...")

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Split audio into segments
        try:
            segment_files = split_audio_to_segments(file_path, segment_duration, temp_folder)
        except RuntimeError as e:
            print(f"Error splitting {file_path}: {e}")
            continue

        results = []
        for i, segment_path in enumerate(tqdm(segment_files, desc=f"Processing Segments for {file_name}", leave=False)):
            try:
                # Read segment audio
                data, sr = sf.read(segment_path, dtype="float32")
                max_val = np.max(np.abs(data))
                if max_val > 0:
                    data /= max_val  # Normalize to [-1.0, 1.0]

                # Classify with YamNet
                scores, _ = classify_audio_segment(data, sr, yamnet_model)

                # Get top N predictions
                avg_scores = scores.mean(axis=0)
                top_indices = np.argsort(avg_scores)[-top_n:][::-1]
                top_classes = [class_names[idx] for idx in top_indices]
                top_scores = avg_scores[top_indices]

                # Save result row
                start_sec = i * segment_duration
                end_sec = start_sec + segment_duration
                row = {"Start": start_sec, "End": end_sec}
                for rank, (class_name, score) in enumerate(zip(top_classes, top_scores), start=1):
                    row[f"Label {rank}"] = class_name
                    row[f"Label {rank} Prob"] = float(score)

                results.append(row)
            except Exception as e:
                print(f"Error processing segment {segment_path}: {e}")

        # Write results to CSV
        csv_filename = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.csv")
        fieldnames = ["Start", "End"] + [f"Label {i}" for i in range(1, top_n + 1)] + [f"Label {i} Prob" for i in range(1, top_n + 1)]
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Results saved to {csv_filename}")

        # Cleanup temporary segment files
        for segment_path in segment_files:
            os.remove(segment_path)
        os.rmdir(temp_folder)

if __name__ == "__main__":
    folder_path = "film_audios2/"
    output_folder = "classification_tables2"

    # folder_path = "test_experiment/film_audios/"
    # output_folder = "test_experiment/classification_tables"

    yamnet_model, class_names = load_yamnet_model()
    process_audio(folder_path, yamnet_model, class_names, output_folder, top_n=5)
