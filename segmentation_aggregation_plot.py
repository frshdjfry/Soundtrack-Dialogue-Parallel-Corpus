import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib import cm
from matplotlib.colors import ListedColormap


def load_data(file_path):
    """Load and preprocess the main dataset (predictions)."""
    data = pd.read_csv(file_path)
    label_columns = [col for col in data.columns if 'Label' in col and 'Prob' not in col]
    prob_columns = [col for col in data.columns if 'Prob' in col]

    # Extract only "Music" related information
    extracted_data = []
    for _, row in data.iterrows():
        for rank, (label_col, prob_col) in enumerate(zip(label_columns, prob_columns), start=1):
            if row[label_col] == "Music":
                extracted_data.append((
                    int(row['Segment'].split('_')[1].split('.')[0]) * 2,  # Time in seconds
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


def load_aggregated_segments(aggregated_path):
    """Load aggregated segments CSV which contains start, end, and avg_prob of music segments."""
    return pd.read_csv(aggregated_path)


def add_aggregated_row(heatmap_data, time_bins, aggregated_segments):
    """Add a row to the heatmap data representing the aggregated music segments."""
    aggregated_row = []
    for bin_time in time_bins:
        # Find the segments that cover this particular bin_time
        matching_segments = aggregated_segments[
            (aggregated_segments['start'] <= bin_time) & (aggregated_segments['end'] > bin_time)
        ]
        if not matching_segments.empty:
            # If multiple segments overlap, take the average of their avg_prob
            val = matching_segments['avg_prob'].mean()
        else:
            val = 0.0
        aggregated_row.append(val)

    # Add this aggregated row at the bottom
    heatmap_data.loc['Aggregated'] = aggregated_row
    return heatmap_data


def plot_heatmap(dataframe, bin_size, title, xlabel, ylabel, x_tick_interval, aggregated_segments=None, time_limit=None):
    """Plot a heatmap for the given dataframe, including an aggregated row from the segments if provided."""
    # Optional time filter
    if time_limit:
        dataframe = dataframe[dataframe['Time'] <= time_limit]

    # Bin the time axis
    dataframe['Time Bin'] = (dataframe['Time'] // bin_size) * bin_size

    # Pivot table for heatmap data (averaging probabilities by bin and rank)
    max_time = time_limit if time_limit else dataframe['Time'].max()
    time_range = np.arange(0, max_time + bin_size, bin_size)
    heatmap_data = dataframe.pivot_table(
        index='Rank', columns='Time Bin', values='Probability', aggfunc='mean'
    ).reindex(columns=time_range, fill_value=0)

    # Sort ranks descending (Rank 1 at the top)
    heatmap_data = heatmap_data.sort_index(ascending=False)

    # If aggregated segments are provided, add them as an additional row
    if aggregated_segments is not None:
        time_bins = heatmap_data.columns
        heatmap_data = add_aggregated_row(heatmap_data, time_bins, aggregated_segments)

    # Create a custom colormap that starts with white at the lowest probability
    reds = cm.get_cmap('Reds', 256)
    new_colors = reds(np.linspace(0, 1, 256))
    new_colors[0] = np.array([1, 1, 1, 1])  # Set the first color (lowest value) to white
    custom_cmap = ListedColormap(new_colors)

    # Plot the heatmap
    plt.figure(figsize=(20, 8))

    # Y labels: ranks plus 'Aggregated' if present
    yticklabels = [f"Rank {i}" for i in reversed(range(1, 6))]
    if aggregated_segments is not None:
        yticklabels = yticklabels + ['Aggregated']

    ax = sns.heatmap(
        heatmap_data, cmap=custom_cmap, annot=False, cbar_kws={'label': 'Average Probability of Music'},
        xticklabels=x_tick_interval, yticklabels=yticklabels
    )
    print(heatmap_data)
    # Update x-axis labels (convert bin times to appropriate units)
    time_bins = heatmap_data.columns
    if bin_size < 60:
        x_labels = [f"{int(bin_time)}" for bin_time in time_bins]
    else:
        # If using minutes, convert seconds to minutes for readability
        x_labels = [f"{int(bin_time // 60)}" for bin_time in time_bins]

    ax.set_xticks(np.arange(len(time_bins)))
    ax.set_xticklabels(x_labels, fontsize=5, rotation=45)

    # Set titles and labels
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.yticks(fontsize=12)

    # Save figure
    # plt.show()
    plt.savefig(title + '.png', dpi=300, bbox_inches='tight')
    # plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate heatmaps for 'Music' label probabilities with aggregated segments.")
    parser.add_argument("file_path", help="Path to the segment predictions CSV file.")
    parser.add_argument("aggregated_path", help="Path to the aggregated segments CSV file.")
    parser.add_argument("--bin_size", type=int, default=60, help="Size of the time bins in seconds.")
    args = parser.parse_args()

    file_path = args.file_path
    aggregated_path = args.aggregated_path
    bin_size = args.bin_size

    film_name = extract_film_name(file_path)
    enhanced_large_df = load_data(file_path)
    aggregated_segments = load_aggregated_segments(aggregated_path)

    # Plot the heatmap for the entire duration (full length)
    # No second plot for the first 10 minutes, as requested.
    plot_heatmap(
        enhanced_large_df, bin_size=bin_size, aggregated_segments=aggregated_segments,
        title=f'Heatmap of "Music" Label Probabilities \n{film_name}',
        xlabel='Time (minutes)' if bin_size >= 60 else 'Time (seconds)',
        ylabel='Rank (Label Position)', x_tick_interval=10
    )


if __name__ == "__main__":
    main()
