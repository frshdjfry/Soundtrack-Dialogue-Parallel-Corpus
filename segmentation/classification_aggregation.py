import os
import subprocess

import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm



def load_predictions(file_path):
    """Load and preprocess the segment predictions CSV file."""
    data = pd.read_csv(file_path)
    return data

def get_music_prob(row):
    """
    Returns the probability of 'Music' in the given row by scanning
    Label i columns. If 'Music' isn't present, returns 0.0.
    """
    for i in range(1, 6):
        label_col = f"Label {i}"
        prob_col = f"Label {i} Prob"

        if label_col in row and prob_col in row:
            if row[label_col] == "Music":
                return float(row[prob_col])  # Ensure it's float
    return 0.0



def extract_music_from_file(
    csv_path,
    input_aac_path,
    output_folder="extracted_music"
):
    """
    Reads the final CSV table (with 'soundtrack' column),
    extracts each soundtrack as a single AAC file using the defined start
    and end times from the original AAC file, without re-encoding.

    Args:
        csv_path (str): Path to the CSV file (which has 'soundtrack' column).
        input_aac_path (str): Path to the original AAC file.
        output_folder (str): Folder where extracted .aac files are saved.

    Returns:
        dict: A dictionary mapping { soundtrack_id: output_file_path },
              so you can see which file was created for each ID.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the CSV
    df = pd.read_csv(csv_path)

    # Group rows by soundtrack ID (excluding ID 0 which is non-music)
    grouped = df[df["soundtrack"] != 0].groupby("soundtrack")
    # Dictionary to track output file paths for each soundtrack ID
    soundtrack_files = {}
    i = 0
    for soundtrack_id, group in grouped:
        # Determine the overall start and end time for this soundtrack ID
        start_time = group["Start"].min()  # Smallest start time
        end_time = group["End"].max()      # Largest end time
        # Build the output file name and path
        out_filename = f"soundtrack_{soundtrack_id}.aac"
        out_path = os.path.join(output_folder, out_filename)

        # Use ffmpeg to extract this time range without re-encoding
        command = [
            "ffmpeg",
            "-i", input_aac_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c", "copy",  # Copy mode: no re-encoding
            "-y",          # Overwrite without confirmation
            out_path
        ]

        # Run the command
        subprocess.run(command, check=True)

        # Track the output file
        soundtrack_files[soundtrack_id] = out_path

    return soundtrack_files


def update_table_with_soundtrack(music_segments, csv_path, output_csv=None):
    """
    Reads a CSV table containing columns:
      - 'Start' (float seconds)
      - 'End'   (float seconds)
      - Possibly 'Label i' / 'Label i Prob' columns (1..5)
    Adds a 'soundtrack' column indicating which recognized music segment
    each row belongs to (by ID). Rows that do not overlap any music segment
    are marked with 0.

    Args:
        music_segments (list of dict): Output from `extract_music_segments`.
            Each dict has 'start', 'end', 'avg_prob'.
        csv_path (str): Path to the CSV file to be updated.
        output_csv (str or None): Optional path for saving the updated CSV.
                                  If None, overwrites `csv_path`.

    Returns:
        pd.DataFrame: The updated dataframe with the 'soundtrack' column.
    """
    # Read CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Initialize a new 'soundtrack' column to 0 (meaning "no music segment")
    df["soundtrack"] = 0

    # Define a small helper to check overlap (partial overlap)
    def is_overlapping(start1, end1, start2, end2):
        """Returns True if [start1, end1] and [start2, end2] overlap at all."""
        return not (end1 <= start2 or start1 >= end2)

    # For each row, check if it overlaps any music segment
    for idx, row in df.iterrows():
        row_start = float(row["Start"])
        row_end   = float(row["End"])

        # Find which music segment (if any) overlaps this row
        # We'll assign the first segment ID that overlaps.
        for i, seg in enumerate(music_segments, start=1):
            seg_start = seg["start"]
            seg_end   = seg["end"]

            # Check for partial overlap
            if is_overlapping(row_start, row_end, seg_start, seg_end):
                df.at[idx, "soundtrack"] = i
                break  # Stop after the first matching segment

    # Decide where to save the updated CSV
    if output_csv is None:
        output_csv = csv_path  # Overwrite the original by default

    # Write back to CSV
    df.to_csv(output_csv, index=False)

    return df

def extract_music_segments(
        data,
        threshold=0.4,
        patience_segments=1,
        min_length=14
):
    """
    Extracts music segments (based on 'Music' probability) that exceed a given
    threshold, allowing for a certain patience of consecutive non-music segments.

    Parameters:
    - data: Pandas DataFrame with columns including "Start", "End",
            "Label i", "Label i Prob" (for i in 1..5).
    - threshold: Probability threshold for considering a row 'Music'.
    - patience_segments: Number of consecutive non-music segments to allow
                         before finalizing the current music region.
    - min_length: Minimum total length (in seconds) for a valid music segment.

    Returns:
    A list of dictionaries, each with:
      {
        "start": float,
        "end": float,
        "avg_prob": float
      }
    """
    music_segments = []

    # Current in-progress music segment
    current_segment = {
        "start": None,
        "end": None,
        "probs": []  # store a list of probabilities for averaging
    }

    # Track the non-music segments we might allow inside a music segment
    pending_non_music = []
    current_patience = patience_segments

    for idx, row in data.iterrows():
        # Identify the start & end time directly from the DataFrame
        start_time = float(row["Start"])
        end_time = float(row["End"])

        # Find the 'Music' probability if present
        music_prob = get_music_prob(row)
        is_music = (music_prob >= threshold)

        if is_music:
            # If this segment is music
            if current_segment["start"] is None:
                # Start a new music segment
                current_segment["start"] = start_time
                current_segment["end"] = end_time
                current_segment["probs"] = [music_prob]
            else:
                # We are already in a music segment or bridging with pending non-music
                # If there were pending non-music segments, incorporate them
                if pending_non_music:
                    for pending in pending_non_music:
                        current_segment["end"] = pending["end"]
                        current_segment["probs"].append(pending["prob"])

                    # Clear pending non-music data
                    pending_non_music = []
                    current_patience = patience_segments

                # Extend the current segment with this music row
                current_segment["end"] = end_time
                current_segment["probs"].append(music_prob)

        else:
            # This row is non-music
            if current_segment["start"] is not None:
                # We are in the middle of a music segment or bridging
                if current_patience > 0:
                    # Temporarily store this non-music segment
                    pending_non_music.append({
                        "start": start_time,
                        "end": end_time,
                        "prob": music_prob  # possibly 0.0 or a small number
                    })
                    current_patience -= 1
                else:
                    # Patience exhausted → finalize the current segment
                    segment_length = current_segment["end"] - current_segment["start"]
                    if segment_length >= min_length:
                        avg_prob = sum(current_segment["probs"]) / len(current_segment["probs"])
                        music_segments.append({
                            "start": current_segment["start"],
                            "end": current_segment["end"],
                            "avg_prob": avg_prob
                        })

                    # Reset current segment & patience
                    current_segment = {
                        "start": None,
                        "end": None,
                        "probs": []
                    }
                    pending_non_music = []
                    current_patience = patience_segments

    # After the loop: check if there's a trailing segment
    if current_segment["start"] is not None:
        # Some pending non-music segments may remain. If the user wants to
        # keep them as part of the final segment, you could incorporate them here
        # if you haven't already decided to finalize them. But typically,
        # if we ended without more music, we just finalize what we have.
        segment_length = current_segment["end"] - current_segment["start"]
        if segment_length >= min_length:
            avg_prob = sum(current_segment["probs"]) / len(current_segment["probs"])
            music_segments.append({
                "start": current_segment["start"],
                "end": current_segment["end"],
                "avg_prob": avg_prob
            })

    return music_segments






if __name__ == "__main__":
    classification_tables_path = 'classification_tables2'
    film_audios_path = 'film_audios2'
    soundtrack_path = 'soundtracks'

    # classification_tables_path = 'test_experiment/classification_tables'
    # film_audios_path = 'test_experiment/film_audios'
    # soundtrack_path = 'test_experiment/soundtracks'

    for file_name in tqdm(os.listdir(classification_tables_path), desc="Processing Files"):
        file_path = os.path.join(classification_tables_path, file_name)
        print(f"Processing {file_path}...")


        # Load predictions
        data = load_predictions(file_path)

        # Extract music segments with patience and minimum length conditions
        music_segments = extract_music_segments(data)
        update_table_with_soundtrack(music_segments, file_path)
        extract_music_from_file(
            file_path,
            os.path.join(film_audios_path, file_name.replace('csv', 'aac')),
            os.path.join(soundtrack_path, file_name.replace('.csv', ''))
        )

