import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def load_data(file_path):
    """Load and preprocess the main dataset."""
    data = pd.read_csv(file_path)
    label_columns = [col for col in data.columns if 'Label' in col and 'Prob' not in col]
    prob_columns = [col for col in data.columns if 'Prob' in col]

    # Extract only "Music" related information
    extracted_data = []
    for _, row in data.iterrows():
        for rank, (label_col, prob_col) in enumerate(zip(label_columns, prob_columns), start=1):
            if row[label_col] == "Music":
                extracted_data.append((
                    int(row['Segment'].split('_')[1].split('.')[0]) * 2,  # Time (x-axis)
                    rank,  # Rank position (y-axis)
                    row[prob_col]  # Probability (color/size)
                ))
    return pd.DataFrame(extracted_data, columns=['Time', 'Rank', 'Probability'])


def extract_film_name(file_path):
    """Extract the film name from the file path."""
    file_name = os.path.basename(file_path)
    # Remove the prefix "segment_predictions_" and the suffix ".csv"
    if file_name.startswith("segment_predictions_"):
        file_name = file_name[len("segment_predictions_"):]
    film_name = os.path.splitext(file_name)[0]
    return film_name.replace('_', ' ')  # Replace underscores with spaces for readability

def load_ground_truth(ground_truth_path):
    """Load ground truth data with start, end, and volume."""
    return pd.read_csv(ground_truth_path)


def add_ground_truth_row(heatmap_data, time_bins, ground_truth):
    """Add a ground truth row to the heatmap data, using volume as the probability."""
    ground_truth_row = []
    for bin_time in time_bins:
        # Find the corresponding volume for the bin if within a ground truth interval
        matching_intervals = ground_truth[
            (ground_truth['start'] <= bin_time) & (ground_truth['end'] > bin_time)
        ]
        if not matching_intervals.empty:
            # Use the average volume if multiple intervals overlap (unlikely but possible)
            volume = matching_intervals['volume'].mean()
        else:
            volume = 0.0  # No music in this bin
        ground_truth_row.append(volume)

    # Add the ground truth row as the last row in the heatmap data
    heatmap_data.loc['Ground Truth'] = ground_truth_row
    return heatmap_data


def plot_heatmap(dataframe, bin_size, title, xlabel, ylabel, x_tick_interval, time_limit=None, ground_truth=None):
    """Plot a heatmap for the given dataframe, including a ground truth row."""
    # Optional time filter
    if time_limit:
        dataframe = dataframe[dataframe['Time'] <= time_limit]

    # Bin time
    dataframe['Time Bin'] = (dataframe['Time'] // bin_size) * bin_size

    # Pivot table for heatmap data
    heatmap_data = dataframe.pivot_table(
        index='Rank', columns='Time Bin', values='Probability', aggfunc='mean'
    ).reindex(columns=np.arange(0, (time_limit or dataframe['Time'].max()) + bin_size, bin_size), fill_value=0)

    heatmap_data = heatmap_data.sort_index(ascending=False)

    # Add the ground truth row if provided
    if ground_truth is not None:
        time_bins = heatmap_data.columns
        heatmap_data = add_ground_truth_row(heatmap_data, time_bins, ground_truth)

    # Use the "Reds" colormap that fades to white for zero probabilities
    reds = cm.get_cmap('Reds', 256)
    new_colors = reds(np.linspace(0, 1, 256))
    new_colors[0] = np.array([1, 1, 1, 1])  # Set the first color (lowest value) to white
    custom_cmap = ListedColormap(new_colors)

    # Plot the heatmap
    plt.figure(figsize=(20, 8))
    yticklabels = [f"Rank {i}" for i in reversed(range(1, 6))]
    if ground_truth is not None:
        yticklabels = yticklabels + ['Ground Truth']
    ax = sns.heatmap(
        heatmap_data, cmap=custom_cmap, annot=False, cbar_kws={'label': 'Average Probability of Music'},
        xticklabels=x_tick_interval, yticklabels=yticklabels
    )

    # Update x-axis labels
    time_bins = heatmap_data.columns
    x_labels = [f"{int(bin_time)}" if bin_size < 60 else f"{int(bin_time // 60)}" for bin_time in time_bins]
    ax.set_xticks(np.arange(len(time_bins)))
    ax.set_xticklabels(x_labels, fontsize=5, rotation=45)

    # Update titles and labels
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.yticks(fontsize=12)
    # plt.show()
    plt.savefig(title + '.png', dpi=300, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description="Generate heatmaps for 'Music' label probabilities.")
    parser.add_argument("file_path", help="Path to the segment predictions CSV file.")
    parser.add_argument("ground_truth_path", help="Path to the ground truth CSV file.")
    args = parser.parse_args()

    # Load and process the data
    file_path = args.file_path
    ground_truth_path = args.ground_truth_path
    film_name = extract_film_name(file_path)
    enhanced_large_df = load_data(file_path)
    ground_truth = load_ground_truth(ground_truth_path)

    # Plot the heatmap for the entire duration (1-minute bins)
    plot_heatmap(
        enhanced_large_df, bin_size=60, ground_truth=None,
        title=f'Heatmap of "Music" Label Probabilities Across Time and Rank\n{film_name}',
        xlabel='Time (minutes)', ylabel='Rank (Label Position)', x_tick_interval=10
    )

    # Plot the heatmap for the first 10 minutes (10-second bins)
    plot_heatmap(
        enhanced_large_df, bin_size=10, time_limit=600, ground_truth=ground_truth,
        title=f'Heatmap of "Music" Label Probabilities (First 10 Minutes, 10-Second Bins)\n{film_name}',
        xlabel='Time (seconds)', ylabel='Rank (Label Position)', x_tick_interval=1
    )

if __name__ == "__main__":
    main()