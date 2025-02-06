import os
import csv
import subprocess

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

###############################################################################
# 1) HARDCODE TARGET CATEGORIES
###############################################################################
MUSICAL_GENRES = [
    'Pop music', 'Hip hop music', 'Rock music', 'Rhythm and blues', 'Soul music', 'Reggae',
    'Country', 'Funk', 'Folk music', 'Middle Eastern music', 'Jazz', 'Disco', 'Classical music',
    'Electronic music', 'Music of Latin America', 'Blues', 'Music for children', 'New-age music',
    'Vocal music', 'Music of Africa', 'Christian music', 'Music of Asia', 'Ska', 'Traditional music',
    'Independent music'
]

MUSICAL_INSTRUMENTS = [
    'Plucked string instrument', 'Keyboard (musical)', 'Percussion', 'Orchestra', 'Brass instrument',
    'Bowed string instrument', 'Wind instrument, woodwind instrument', 'Harp', 'Choir', 'Bell',
    'Harmonica', 'Accordion', 'Bagpipes', 'Didgeridoo', 'Shofar', 'Theremin', 'Singing bowl',
    'Musical ensemble', 'Bass (instrument role)', 'Scratching (performance technique)'
]

MOODS = [
    'Happy music', 'Funny music', 'Sad music', 'Tender music', 'Exciting music',
    'Angry music', 'Scary music'
]


###############################################################################
# 2) LOAD YAMNET MODEL AND CLASS NAMES
###############################################################################
def load_yamnet_model():
    """
    Loads the YAMNet model from TensorFlow Hub and retrieves class labels.
    """
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
            class_names.append(row[2])  # Third column contains human-readable class names
    return model, class_names


###############################################################################
# 3) HELPER FUNCTIONS FOR AUDIO PROCESSING
###############################################################################
def get_audio_duration(file_path):
    """
    Uses ffprobe to get the duration of an audio file in seconds.
    """
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(result.stdout.strip()) if result.stdout.strip() else 0.0
    except ValueError:
        return 0.0  # Default to 0.0 if parsing fails


def convert_aac_to_wav(aac_path, wav_path):
    """
    Converts .aac to a 16 kHz mono .wav file using ffmpeg.
    """
    command = [
        "ffmpeg", "-y",  # Overwrite output if exists
        "-i", aac_path,  # Input file
        "-ar", "16000",  # Resample to 16 kHz
        "-ac", "1",  # Mono
        wav_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def classify_audio_file(audio_path, model):
    """
    Loads the WAV file from 'audio_path', processes it with YAMNet,
    and returns the average scores for each class.
    """
    file_contents = tf.io.read_file(audio_path)
    audio, _ = tf.audio.decode_wav(file_contents, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)  # Flatten audio data

    # Normalize to [-1.0, 1.0] based on max absolute value
    max_val = np.max(np.abs(audio.numpy()))
    if max_val > 0:
        audio = audio / max_val

    # Run inference
    scores, embeddings, spectrogram = model(audio)
    mean_scores = tf.reduce_mean(scores, axis=0)  # Average across time frames
    return mean_scores.numpy()


###############################################################################
# 4) PROCESS AUDIO FILES AND STORE RESULTS
###############################################################################
def process_aac_folders(root_folder, output_folder, model, yamnet_classes):
    """
    Iterates over IMDb ID subfolders, processes AAC files, converts to WAV, classifies with YAMNet,
    and stores probabilities separately for genres, instruments, and moods, along with the duration.
    """
    # Build mappings for each category type
    genre_to_index = {name: yamnet_classes.index(name) for name in MUSICAL_GENRES if name in yamnet_classes}
    instrument_to_index = {name: yamnet_classes.index(name) for name in MUSICAL_INSTRUMENTS if name in yamnet_classes}
    mood_to_index = {name: yamnet_classes.index(name) for name in MOODS if name in yamnet_classes}

    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over IMDb folders
    for imdb_id in tqdm(os.listdir(root_folder), desc="Processing Soundtracks"):
        subfolder = os.path.join(root_folder, imdb_id)
        if not os.path.isdir(subfolder):
            continue

        results = []  # Store rows as [soundtrack_id, duration, genre_probs..., instrument_probs..., mood_probs...]

        # Process each AAC file in the IMDb folder
        for fname in os.listdir(subfolder):
            if fname.endswith(".aac"):
                soundtrack_id = os.path.splitext(fname)[0]  # Extract soundtrack ID (e.g., "1" from "1.aac")
                aac_path = os.path.join(subfolder, fname)

                # Get duration of the original AAC file
                duration = get_audio_duration(aac_path)

                # Convert to temporary WAV file
                wav_path = os.path.join(subfolder, f"{soundtrack_id}_temp.wav")
                convert_aac_to_wav(aac_path, wav_path)

                # Classify with YAMNet
                mean_scores = classify_audio_file(wav_path, model)

                # Remove temp file
                if os.path.exists(wav_path):
                    os.remove(wav_path)

                # Extract probabilities for genres, instruments, and moods
                genre_probs = [mean_scores[genre_to_index[name]] for name in genre_to_index]
                instrument_probs = [mean_scores[instrument_to_index[name]] for name in instrument_to_index]
                mood_probs = [mean_scores[mood_to_index[name]] for name in mood_to_index]

                # Store results in a row (including duration)
                row = [soundtrack_id, duration] + genre_probs + instrument_probs + mood_probs
                results.append(row)

        # Write CSV for this IMDb ID
        if results:
            # Prepare CSV headers
            header = ["soundtrack_id", "duration"] + list(genre_to_index.keys()) + list(
                instrument_to_index.keys()) + list(mood_to_index.keys())
            out_csv_path = os.path.join(output_folder, f"{imdb_id}.csv")

            with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(results)

            # print(f"Saved results for IMDb {imdb_id} to {out_csv_path}.")


###############################################################################
# 5) SCRIPT ENTRY POINT
###############################################################################
if __name__ == "__main__":
    # Define input and output folder paths
    root_folder_with_imdb = "./soundtracks"  # Path to the folder containing IMDb folders
    output_csv_folder = "./soundtracks_features"  # Where CSVs will be saved

    # Load YAMNet model and its class labels
    yamnet_model, yamnet_class_names = load_yamnet_model()

    # Process all audio files and store results
    process_aac_folders(
        root_folder=root_folder_with_imdb,
        output_folder=output_csv_folder,
        model=yamnet_model,
        yamnet_classes=yamnet_class_names
    )
