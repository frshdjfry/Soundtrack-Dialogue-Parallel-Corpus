import os
import re
import csv
import moviepy.editor as mp
from pydub import AudioSegment
import soundfile as sf
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def natural_sort_key(filename):
    """
    Extracts the numerical part of the filename for natural sorting.
    """
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

# Step 1: Extract audio from video
def extract_audio_from_video(mp4_path, output_audio_path):
    video = mp.VideoFileClip(mp4_path)
    video.audio.write_audiofile(output_audio_path)

# Step 2: Split audio into 1-second segments
def split_audio_into_seconds(audio_path, output_dir):
    audio = AudioSegment.from_file(audio_path)
    one_second_ms = 2000  # 1000 milliseconds = 1 second
    duration_seconds = len(audio) // one_second_ms

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(duration_seconds):
        segment = audio[i * one_second_ms: (i + 1) * one_second_ms]
        segment.export(os.path.join(output_dir, f"segment_{i}.wav"), format="wav")

# Step 3: Load YAMNet model and class names
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

# Step 4: Classify each audio segment using YAMNet
def classify_audio_segment(segment_path, yamnet_model):
    waveform, sr = sf.read(segment_path)
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != 44100:
        raise ValueError(f"Audio sample rate {sr} Hz is not 16 kHz.")
    scores, embeddings, spectrogram = yamnet_model(waveform)
    predicted_class_indices = tf.argmax(scores, axis=1).numpy()
    return scores.numpy(), predicted_class_indices

# Step 5: Main function to process and classify the whole video
def process_video(mp4_path, yamnet_model, class_names, top_n=3):
    audio_path = "extracted_audio.wav"
    segments_dir = "audio_segments_" + mp4_path
    extract_audio_from_video(mp4_path, audio_path)
    split_audio_into_seconds(audio_path, segments_dir)
    results = []

    for filename in sorted(os.listdir(segments_dir), key=natural_sort_key):
        if filename.endswith(".wav"):
            segment_path = os.path.join(segments_dir, filename)
            scores, predicted_class_indices = classify_audio_segment(segment_path, yamnet_model)
            top_indices = np.argsort(scores.mean(axis=0))[-top_n:][::-1]
            top_classes = [class_names[i] for i in top_indices]
            top_scores = scores.mean(axis=0)[top_indices]

            # Create a row for each segment with segment name, labels, and probabilities
            row = {"Segment": filename}
            for i, (class_name, score) in enumerate(zip(top_classes, top_scores), start=1):
                row[f"Label {i}"] = class_name
                row[f"Label {i} Prob"] = score
            results.append(row)

    # Determine header with dynamic labels and probabilities columns
    fieldnames = ["Segment"]
    for i in range(1, top_n + 1):
        fieldnames.extend([f"Label {i}", f"Label {i} Prob"])

    # Write results to CSV
    with open(f"segment_predictions_{mp4_path}.csv", mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("Results saved to segment_predictions.csv")

if __name__ == "__main__":
    mp4_file = "The_Matrix_Revolutions_2003.mkv"  # Replace with your .mp4 file path
    yamnet_model, class_names = load_yamnet_model()
    process_video(mp4_file, yamnet_model, class_names, top_n=5)  # You can change top_n for more labels
