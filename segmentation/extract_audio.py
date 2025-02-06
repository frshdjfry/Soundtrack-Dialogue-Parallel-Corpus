import os
import subprocess
from pymediainfo import MediaInfo

def extract_audio_from_films(source_folder, target_folder):
    """
    Extracts audio tracks from films in the source folder and saves them in the target folder.
    Prioritizes AAC format if available, and logs problematic files.

    Args:
        source_folder (str): Path to the folder containing movie files.
        target_folder (str): Path to the folder where extracted audio will be saved.
    """
    # Ensure target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # List to track problematic files
    problematic_files = []

    # Loop through all files in the source folder
    for file_name in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file_name)

        # Skip if not a valid file
        if not os.path.isfile(file_path):
            continue

        # Analyze the media file
        media_info = MediaInfo.parse(file_path)
        audio_tracks = [track for track in media_info.tracks if track.track_type == "Audio"]

        if not audio_tracks:
            print(f"No audio track found in: {file_name}")
            problematic_files.append(file_name)
            continue

        # Prioritize AAC format if available, otherwise select the first audio track
        selected_audio_index = None
        for index, track in enumerate(audio_tracks):
            if track.format.lower() == "aac":
                selected_audio_index = index
                break

        # If no AAC track is found, use the first available audio track
        if selected_audio_index is None:
            selected_audio_index = 0

        # Get the selected audio track's format
        selected_audio_track = audio_tracks[selected_audio_index]
        audio_format = selected_audio_track.format.lower()

        print(f"Extracting audio from {file_name} (Format: {audio_format}, Stream Index: {selected_audio_index})")

        # Set output file name
        output_file_name = f"{os.path.splitext(file_name)[0]}.{audio_format}"
        output_file_path = os.path.join(target_folder, output_file_name)

        # Use ffmpeg to extract the audio
        ffmpeg_command = [
            "ffmpeg",
            "-i", file_path,
            "-map", f"0:a:{selected_audio_index}",  # Map the correct audio stream
            "-vn",  # No video
            "-acodec", "copy",  # Copy audio without re-encoding
            output_file_path
        ]

        try:
            subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Audio saved to: {output_file_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio from {file_name}: {e.stderr.decode()}")
            problematic_files.append(file_name)

    # Print problematic files
    if problematic_files:
        print("\nProblematic files:")
        for file in problematic_files:
            print(f" - {file}")

if __name__ == "__main__":
    #
    source_folder = '/Volumes/ADATA HD680/thesis films/films6'
    target_folder = 'film_audios2'
    #
    # source_folder = 'test_experiment/films'
    # target_folder = 'test_experiment/film_audios'

    # Run the extraction process
    extract_audio_from_films(source_folder, target_folder)
