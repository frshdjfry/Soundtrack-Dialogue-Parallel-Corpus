import os
import pandas as pd
import argparse
from pydub import AudioSegment


def load_predictions(file_path):
    """Load and preprocess the segment predictions CSV file."""
    data = pd.read_csv(file_path)
    return data


def extract_music_segments(data, threshold=0.4, patience_segments=1, min_length=14):
    """
    Extract music segments that exceed a certain probability threshold, allowing for a given
    patience of non-music segments to appear in between.

    Parameters:
    - data: Pandas DataFrame of segments with columns ["Segment", "Label 1", "Label 1 Prob"]
    - threshold: Probability threshold for considering a segment as music.
    - patience_segments: How many consecutive non-music segments to allow before finalizing a segment.
    - min_length: Minimum length in seconds for a music segment to be considered valid.

    Returns:
    A list of dictionaries with start time, end time, filenames, and average probability.
    """
    music_segments = []
    current_segment = {
        "start": None,
        "end": None,
        "files": [],
        "probs": []
    }

    # Variables to track patience
    pending_non_music = []  # Will hold non-music segments temporarily
    current_patience = patience_segments

    for idx, row in data.iterrows():
        segment_name = row["Segment"]
        segment_time = int(segment_name.split('_')[1].split('.')[0]) * 2

        is_music = (row["Label 1"] == "Music") and (row["Label 1 Prob"] >= threshold)

        if is_music:
            # If this is a music segment
            if current_segment["start"] is None:
                # Start a new segment
                current_segment["start"] = segment_time
                current_segment["end"] = segment_time + 2
                current_segment["files"] = [segment_name]
                current_segment["probs"] = [row["Label 1 Prob"]]
            else:
                # If we had some pending non-music segments that occurred before this,
                # since we are now continuing music, we include them as well.
                if pending_non_music:
                    # Include the pending non-music segments in the chain
                    for p_seg in pending_non_music:
                        current_segment["end"] = p_seg["end"]
                        current_segment["files"].append(p_seg["segment_name"])
                        current_segment["probs"].append(p_seg["prob"])

                    # Clear pending non-music and reset patience
                    pending_non_music = []
                    current_patience = patience_segments

                # Add current music segment
                current_segment["end"] = segment_time + 2
                current_segment["files"].append(segment_name)
                current_segment["probs"].append(row["Label 1 Prob"])
        else:
            # Not music
            if current_segment["start"] is not None:
                # We are in the middle of a segment, let's see if we can tolerate this gap
                if current_patience > 0:
                    # Temporarily add this non-music segment to pending list
                    pending_non_music.append({
                        "start": segment_time,
                        "end": segment_time + 2,
                        "segment_name": segment_name,
                        "prob": row["Label 1 Prob"]
                    })
                    current_patience -= 1
                else:
                    # No patience left. This means we must finalize the current segment.
                    # First, finalize only if it meets the min_length
                    if current_segment["start"] is not None:
                        segment_length = current_segment["end"] - current_segment["start"]
                        if segment_length >= min_length:
                            # Calculate avg probability
                            avg_prob = sum(current_segment["probs"]) / len(current_segment["probs"])
                            music_segments.append({
                                "start": current_segment["start"],
                                "end": current_segment["end"],
                                "files": current_segment["files"],
                                "avg_prob": avg_prob,
                            })

                    # Reset current segment
                    current_segment = {
                        "start": None,
                        "end": None,
                        "files": [],
                        "probs": []
                    }
                    # Also reset patience and pending
                    pending_non_music = []
                    current_patience = patience_segments

    # After the loop, we may have a trailing segment
    # Check if we still have a valid segment to finalize
    if current_segment["start"] is not None:
        # It's possible we ended with some pending non-music segments that never got resolved back into music.
        # If we have pending_non_music segments at the end that were never followed by music, we discard them.
        # Just finalize the current segment as-is (without pending).
        segment_length = current_segment["end"] - current_segment["start"]
        if segment_length >= min_length:
            avg_prob = sum(current_segment["probs"]) / len(current_segment["probs"])
            music_segments.append({
                "start": current_segment["start"],
                "end": current_segment["end"],
                "files": current_segment["files"],
                "avg_prob": avg_prob,
            })

    return music_segments


def concatenate_audio_files(segments, input_folder, output_folder):
    """
    Concatenate audio files for each segment and save them in the output folder.
    Returns a list of dictionaries summarizing the segments.
    """
    os.makedirs(output_folder, exist_ok=True)
    summary = []

    for i, segment in enumerate(segments):
        combined = AudioSegment.empty()
        for file in segment["files"]:
            file_path = os.path.join(input_folder, file)
            audio = AudioSegment.from_wav(file_path)
            combined += audio

        output_path = os.path.join(output_folder, f"music_segment_{i + 1}.wav")
        combined.export(output_path, format="wav")

        summary.append({
            "start": segment["start"],
            "end": segment["end"],
            "avg_prob": segment["avg_prob"],
            "output_file": os.path.basename(output_path),
            "input_files": ", ".join(segment["files"]),
        })

    return summary


def save_summary(summary, output_csv):
    """Save the summary data to a CSV file."""
    pd.DataFrame(summary).to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Process and combine audio segments with 'Music' labels.")
    parser.add_argument("csv_file", help="Path to the segment predictions CSV file.")
    parser.add_argument("audio_folder", help="Path to the folder containing segmented WAV files.")
    parser.add_argument("output_folder", help="Path to the folder where combined audio will be saved.")
    parser.add_argument("output_csv", help="Path to save the summary CSV file.")
    parser.add_argument("--threshold", type=float, default=0.4, help="Probability threshold for music segments.")
    parser.add_argument("--patience", type=int, default=1, help="Number of consecutive non-music segments to allow.")
    parser.add_argument("--min_length", type=int, default=14, help="Minimum length in seconds for a music segment.")

    args = parser.parse_args()

    # Load predictions
    data = load_predictions(args.csv_file)

    # Extract music segments with patience and minimum length conditions
    music_segments = extract_music_segments(data, threshold=args.threshold,
                                            patience_segments=args.patience,
                                            min_length=args.min_length)

    # Concatenate audio files and save them
    summary = concatenate_audio_files(music_segments, args.audio_folder, args.output_folder)

    # Save summary CSV
    save_summary(summary, args.output_csv)


if __name__ == "__main__":
    main()
