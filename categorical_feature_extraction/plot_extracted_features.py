import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
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

def load_and_join_data(summary_csv, film_info_csv):
    """
    Loads the summary table of music durations by IMDb ID
    and the film info table (origin, decade, film genre).
    Joins them on 'imdb_id'.

    :param summary_csv: Path to the CSV with aggregated durations
                       (columns: imdb_id, label1, label2, etc.).
    :param film_info_csv: Path to the CSV with film info
                          (columns: imdb_id, origin, decade, film_genre, etc.).
    :return: A pandas DataFrame with both tables joined.
    """
    df_summary = pd.read_csv(summary_csv)
    df_info = pd.read_csv(film_info_csv)

    # Perform an inner join on imdb_id (only rows present in both tables)
    df_joined = pd.merge(df_summary, df_info, how="inner", on="imdb_id")

    return df_joined


def plot_heatmaps(df_joined,
                  category_labels,
                  category_name,
                  group_cols=("origin", "decade", "film_genre"),
                  duration_threshold=0.0):
    """
    Plots 3 heatmaps (subplots) in a single figure for one category (e.g. Moods):
      1. label vs origin
      2. label vs decade
      3. label vs film_genre

    :param df_joined: The DataFrame containing duration columns +
                      'origin', 'decade', 'film_genre', etc.
    :param category_labels: A list of columns that belong to this category
                            (e.g., MOODS or MUSICAL_GENRES).
    :param category_name: String to use in figure title (e.g. "Moods").
    :param group_cols: Which columns in df_joined to use for grouping
                       (e.g. "origin", "decade", "film_genre").
    :param duration_threshold: Rows (labels) whose total sum < threshold
                               across all groups are removed from the plot.
    """
    valid_labels = [lbl for lbl in category_labels if lbl in df_joined.columns]

    df_joined[valid_labels] = df_joined[valid_labels].div(df_joined['runtime'] * 60, axis=0)
    # print(df_joined[valid_labels])
    # Prepare the figure with 3 subplots (one for each group_col)
    # fig, axes = plt.subplots(1, len(group_cols),
    #                          figsize=(18, 6),  # wide figure for 3 subplots
    #                          sharey=False)

    fig, axes = plt.subplots(1, len(group_cols), figsize=(18, 6), sharey=False)

    for ax, group_col in zip(axes, group_cols):
        if group_col not in df_joined.columns:
            ax.set_title(f"{group_col} not in DataFrame")
            ax.axis("off")
            continue

        # ────────── 2) Sum fractional durations by group  ──────────
        grouped = df_joined.groupby(group_col)[valid_labels].sum()

        # ────────── 3) Transpose and apply threshold  ──────────
        pivot_df = grouped.transpose()
        pivot_df["sum_all"] = pivot_df.sum(axis=1)
        pivot_df = pivot_df[pivot_df["sum_all"] >= duration_threshold]
        pivot_df = pivot_df.sort_values("sum_all", ascending=False)
        pivot_df.drop(columns=["sum_all"], inplace=True)
        pivot_df.columns = [str(col)[:8] for col in pivot_df.columns]
        pivot_df.index = [str(idx)[:16] for idx in pivot_df.index]
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

        # Rotate x-axis tick labels
        # ax.tick_params(axis='x', rotation=45)

        # Rotate y-axis tick labels
        # ax.tick_params(axis='y', rotation=70)

        if pivot_df.empty:
            ax.set_title(f"No {category_name} >= {duration_threshold}")
            ax.axis("off")
            continue

        # ────────── 4) Plot heatmap  ──────────
        sns.heatmap(pivot_df, ax=ax, annot=True, fmt=".2f", cmap="viridis", annot_kws={"size": 6})
        ax.set_title(f"{category_name} by {group_col}")
        # ax.set_xlabel(group_col)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=6)  # Reduce colorbar label size
        cbar.ax.set_aspect(30)  # Make co


    plt.subplots_adjust(wspace=0.7)
    fig.suptitle(f"{category_name}", fontsize=16)
    plt.tight_layout()
    plt.show()


def main_plot():
    # 1) Paths to your CSV files
    summary_csv = "soundtracks_features_summary.csv"  # from your aggregator
    film_info_csv = "titles/top_5_combined.csv"    # your table with origin, decade, film_genre

    # 2) Load & join
    df_joined = load_and_join_data(summary_csv, film_info_csv)

    # 3) Plot for each category:
    #    a) MUSIC GENRE
    plot_heatmaps(df_joined,
                  category_labels=MUSICAL_GENRES,
                  category_name="Music Genres",
                  group_cols=("origin", "decade", "film_genre"),
                  duration_threshold=0.01)

    #    b) INSTRUMENTS
    plot_heatmaps(df_joined,
                  category_labels=MUSICAL_INSTRUMENTS,
                  category_name="Instruments",
                  group_cols=("origin", "decade", "film_genre"),
                  duration_threshold=0.01)

    #    c) MOODS
    plot_heatmaps(df_joined,
                  category_labels=MOODS,
                  category_name="Moods",
                  group_cols=("origin", "decade", "film_genre"),
                  duration_threshold=0.01)

if __name__ == "__main__":
    main_plot()
