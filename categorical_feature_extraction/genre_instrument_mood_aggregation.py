import os
import pandas as pd
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
# 2) AGGREGATE DURATION FOR TOP PREDICTED LABELS
###############################################################################
def aggregate_durations(input_folder, output_file):
    """
    Reads all soundtrack feature CSVs, determines the top predicted label
    for each category per soundtrack, and aggregates the total duration
    per label for each movie.

    :param input_folder: Path to the folder containing per-movie CSVs.
    :param output_file: Path to save the final aggregated table.
    """

    # Initialize final summary data structure
    summary_data = []

    # Process each movie (each CSV in the folder)
    for csv_file in tqdm(os.listdir(input_folder), desc="Processing Movies"):
        if not csv_file.endswith(".csv"):
            continue  # Skip non-CSV files

        imdb_id = os.path.splitext(csv_file)[0]  # Extract IMDb ID from filename
        csv_path = os.path.join(input_folder, csv_file)

        # Load the movie's soundtrack data
        df = pd.read_csv(csv_path)

        # Ensure required columns exist
        if "duration" not in df.columns:
            print(f"Skipping {csv_file} (missing duration column)")
            continue

        # Filter available labels in the dataset (avoid KeyErrors)
        available_genres = [label for label in MUSICAL_GENRES if label in df.columns]
        available_instruments = [label for label in MUSICAL_INSTRUMENTS if label in df.columns]
        available_moods = [label for label in MOODS if label in df.columns]

        # Initialize duration counters for each category
        genre_durations = {label: 0.0 for label in available_genres}
        instrument_durations = {label: 0.0 for label in available_instruments}
        mood_durations = {label: 0.0 for label in available_moods}

        # Iterate over each soundtrack
        for _, row in df.iterrows():
            duration = row["duration"]  # Get soundtrack duration

            # Extract the highest probability label for each category (only if columns exist)
            genre_label = row[available_genres].idxmax() if available_genres and any(
                row[available_genres] > 0) else None
            instrument_label = row[available_instruments].idxmax() if available_instruments and any(
                row[available_instruments] > 0) else None
            mood_label = row[available_moods].idxmax() if available_moods and any(row[available_moods] > 0) else None

            # Accumulate duration for the highest predicted label in each category
            if genre_label:
                genre_durations[genre_label] += duration
            if instrument_label:
                instrument_durations[instrument_label] += duration
            if mood_label:
                mood_durations[mood_label] += duration

        # Combine all results into a single row for this movie
        movie_row = {"imdb_id": imdb_id}
        movie_row.update(genre_durations)
        movie_row.update(instrument_durations)
        movie_row.update(mood_durations)

        summary_data.append(movie_row)

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Save aggregated table
    summary_df.to_csv(output_file, index=False)
    print(f"\nSummary table saved to {output_file}")


###############################################################################
# 3) SCRIPT ENTRY POINT
###############################################################################
if __name__ == "__main__":
    input_csv_folder = "soundtracks_features"  # Folder with per-movie soundtrack tables
    output_csv_path = "soundtracks_features_summary.csv"  # Aggregated output file

    # Process all soundtrack tables and generate the summary
    aggregate_durations(input_csv_folder, output_csv_path)
