import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from typing import Optional
import textwrap
from mplsoccer import VerticalPitch
from math import pi

from src.config import *


def closest_difference(df: pd.DataFrame, column: str) -> list[float]:
    """
    Calculate the closest difference between the values in a column of a DataFrame. The closest difference is calculated by
    finding the minimum absolute difference between the value in a row and all other values in the column. The closest
    difference is calculated for each value in the column and returned as a list.

    Args:
        df (pd.DataFrame): DataFrame containing the column of values
        column (str): The column containing the values to calculate the closest difference for

    Returns:
        closest_diff (list[float]): A list containing the closest difference for each value in the column
    """

    # Extract the values from the column as a numpy array
    values = df[column].values

    # Calculate the closest difference for each value in the column, excluding itself, and store in a list
    closest_diff = []
    for i, val in enumerate(values):
        diffs = values - val
        diffs[i] = np.inf
        closest_diff.append(np.min(diffs))

    # Return the list of closest differences
    return closest_diff


def wrap_label(label: str, width: int) -> str:
    """
    Wrap a label to a specified width by splitting the label into multiple lines. The label is wrapped by splitting the label
    into words and adding each word to a line until the line reaches the specified width. The wrapped label is returned as a
    single string with newline characters separating each line.

    Args:
        label (str): The label to be wrapped
        width (int): The maximum width of each line

    Returns:
        label (str): The wrapped label with newline characters separating each line
    """

    # Wrap the label by splitting the label into words and adding each word to a line until the line reaches the specified width
    label = "\n".join(textwrap.wrap(label, width))

    # Return the wrapped label
    return label


def plot_receptions_on_pitch(receptions):
    """
    Plots proportion of ball receipts in each of the 18 pitch zones for each player in the dataset. Plotting onto a pitch is
    performed using the mplsoccer library and statsbomb pitch template. The pitch is divided into 18 zones by splitting the
    length of the pitch into 6 equal zones and the width into 3 zones using lines at 20 and 60 yards. The proportion of ball
    receipts in each zone is represented by a heatmap with the intensity of the color representing the proportion of receipts
    in that zone. For each player in the dataset, a heatmap is saved as an image file in the data/outputs folder.

    Args:
        receptions (pd.DataFrame): DataFrame containing infromation on each ball receipt
    """

    # Define the path effects for the heatmap labels
    path_eff = [
        path_effects.Stroke(linewidth=3, foreground=BLACK),
        path_effects.Normal(),
    ]

    # Create vertical pitch object
    pitch = VerticalPitch(
        pitch_type=PITCH_TYPE,
        pitch_color=PITCH_COLOUR,
        line_color=PITCH_COLOUR,
        line_zorder=2,
    )

    # Define the pitch zones
    bin_x = np.linspace(pitch.dim.left, pitch.dim.right, num=7)
    bin_y = np.sort(
        np.array([pitch.dim.bottom, Y_ZONE_LINE_MIN, Y_ZONE_LINE_MAX, pitch.dim.top])
    )

    # Find the global maximum proportion of zone receptions to ensure consistency in color scale across all players
    global_max = 0
    for player_id in PLAYER_IDS:
        df = receptions[receptions[PLAYER_ID] == player_id]
        bin_statistic = pitch.bin_statistic(
            df[LOCATION_X],
            df[LOCATION_Y],
            statistic=COUNT,
            bins=(bin_x, bin_y),
            normalize=True,
        )
        global_max = max(global_max, bin_statistic[STATISTIC].max())

    # Plot receptions heatmap for each player and save as an image file in the data/outputs folder
    for player_id in PLAYER_IDS:
        df = receptions[receptions[PLAYER_ID] == player_id]
        fig, ax = pitch.draw(figsize=(12, 8))
        fig.set_facecolor(PITCH_COLOUR)
        bin_statistic = pitch.bin_statistic(
            df[LOCATION_X],
            df[LOCATION_Y],
            statistic=COUNT,
            bins=(bin_x, bin_y),
            normalize=True,
        )
        pitch.heatmap(
            bin_statistic, ax=ax, cmap=BLUES, edgecolor=GREY, zorder=1, vmax=global_max
        )
        labels2 = pitch.label_heatmap(
            bin_statistic,
            color=WHITE,
            fontsize=36,
            ax=ax,
            ha=CENTER,
            va=CENTER,
            str_format="{:.0%}",
            path_effects=path_eff,
            weight=BOLD,
        )
        fig.savefig(
            f"{DATA_OUTPUTS_PATH}{PLAYER}_{player_id}_{RECEPTIONS_HEATMAP_NAME}",
            bbox_inches=TIGHT,
        )


def create_scatter_plot_of_two_metrics(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    colour_mapping: dict,
    size_metric: Optional[str] = None,
    y_x_line: bool = False,
):
    """
    Create a scatter plot of two proposed metrics for each player in the dataset. The scatter plot is created using matplotlib
    and the color of each player's point is determined by the color mapping provided. The size of each point can be scaled by
    a metric, if one is provided. Additionally, a line can be drawn on the plot to represent when the x and y metrics are equal
    to help illustrate conversion rate of goals to expected goals. The scatter plot is saved as an image file in the
    data/outputs folder.

    Args:
        df (pd.DataFrame): DataFrame containing the metrics to be plotted
        x_metric (str): The metric to be plotted on the x-axis
        y_metric (str): The metric to be plotted on the y-axis
        colour_mapping (dict): A dictionary mapping each player to a color
        size_metric (Optional[str]): The metric to be used to scale the size of each point
        y_x_line (bool): A boolean indicating whether a line should be drawn on the plot where x and y metrics are equal
    """

    # Create labels for each point on scatter plot. If TeamID is present, this will be included alonside player name as there
    # may be multiple points for a player as a result of playing for multiple teams
    if TEAM_ID in df.columns:
        df[PLAYER_TEAM_COMBO] = df[PLAYER_NAME] + " (" + df[TEAM_NAME] + ")"
    else:
        df[PLAYER_TEAM_COMBO] = df[PLAYER_NAME]

    # Create a color column based on the color mapping provided
    df[COLOR] = df[PLAYER_NAME].map(colour_mapping)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 9.5))

    # Define limits to support the placing of labels next to player points
    x_min, x_max = df[x_metric].min(), df[x_metric].max()
    y_min, y_max = df[y_metric].min(), df[y_metric].max()
    x_word_length = (x_max - x_min) * 0.25
    y_word_height = (y_max - y_min) * 0.05
    x_lim_max = x_max - x_word_length
    x_lim_min = x_min + x_word_length
    y_lim_max = y_max - y_word_height
    y_lim_min = y_min + y_word_height
    default_x_offset = (x_max - x_min) * 0.02
    default_y_offset = (y_max - y_min) * 0.02

    # Calculate the closest difference between points to avoid overlapping labels next to points
    df[X_DIFF] = closest_difference(df, x_metric)
    df[Y_DIFF] = closest_difference(df, y_metric)

    # Ensure horizontal and vertical alignment will avoid labels being placed outside the plot
    df.loc[df[x_metric] > x_lim_max, HA] = RIGHT
    df.loc[df[x_metric] < x_lim_min, HA] = LEFT
    df.loc[df[y_metric] > y_lim_max, VA] = TOP
    df.loc[df[y_metric] < y_lim_min, VA] = BOTTOM

    # Create columns to store the x and y offsets for each point
    df[X_OFFSET] = None
    df[Y_OFFSET] = None

    # Split the points into potential overlapping and non-overlapping points
    overlapping_points = df[
        (np.abs(df[X_DIFF]) <= 2 * x_word_length)
        & (np.abs(df[Y_DIFF]) <= 2 * y_word_height)
    ]
    non_overlapping_points = df[~df.index.isin(overlapping_points.index)]

    # For potential overlapping points, determine the x and y offsets and the horizontal and vertical alignment to help avoid
    # overlapping labels
    if not overlapping_points.empty:
        overlapping_points.loc[
            (overlapping_points[Y_DIFF] > 0) & (overlapping_points[VA] == TOP),
            Y_OFFSET,
        ] = (
            0 - default_y_offset
        )
        overlapping_points.loc[
            (overlapping_points[Y_DIFF] > 0) & (overlapping_points[VA] == BOTTOM),
            Y_OFFSET,
        ] = 0 - (0.5 * default_y_offset)
        overlapping_points.loc[
            (overlapping_points[Y_DIFF] < 0) & (overlapping_points[VA] == BOTTOM),
            Y_OFFSET,
        ] = default_y_offset
        overlapping_points.loc[
            (overlapping_points[Y_DIFF] < 0) & (overlapping_points[VA] == TOP),
            Y_OFFSET,
        ] = 0 - (0.5 * default_y_offset)
        overlapping_points.loc[
            (overlapping_points[X_DIFF] > 0) & (overlapping_points[HA] == LEFT),
            X_OFFSET,
        ] = (
            0.5 * default_x_offset
        )
        overlapping_points.loc[
            (overlapping_points[X_DIFF] > 0) & (overlapping_points[HA] == RIGHT),
            X_OFFSET,
        ] = (
            0 - default_x_offset
        )
        overlapping_points.loc[
            (overlapping_points[X_DIFF] < 0) & (overlapping_points[HA] == LEFT),
            X_OFFSET,
        ] = default_x_offset
        overlapping_points.loc[
            (overlapping_points[X_DIFF] < 0) & (overlapping_points[HA] == RIGHT),
            X_OFFSET,
        ] = 0 - (0.5 * default_x_offset)
        overlapping_points.loc[
            (overlapping_points[Y_DIFF] > 0) & (overlapping_points[VA].isnull()), VA
        ] = TOP
        overlapping_points.loc[
            (overlapping_points[Y_DIFF] < 0) & (overlapping_points[VA].isnull()), VA
        ] = BOTTOM
        overlapping_points.loc[
            (overlapping_points[X_DIFF] > 0) & (overlapping_points[HA].isnull()), HA
        ] = RIGHT
        overlapping_points.loc[
            (overlapping_points[X_DIFF] < 0) & (overlapping_points[HA].isnull()), HA
        ] = LEFT

    # Combine the non-overlapping and overlapping points
    df = pd.concat([non_overlapping_points, overlapping_points], ignore_index=True)

    # Ensure horizontal and vertical alignment and offsets are set for all points
    df.loc[df[HA].isnull(), HA] = RIGHT
    df.loc[df[VA].isnull(), VA] = TOP
    df.loc[(df[X_OFFSET].isnull()) & (df[HA] == LEFT), X_OFFSET] = default_x_offset
    df.loc[(df[X_OFFSET].isnull()) & (df[HA] == RIGHT), X_OFFSET] = 0 - default_x_offset
    df.loc[(df[Y_OFFSET].isnull()) & (df[VA] == TOP), Y_OFFSET] = 0 - default_y_offset
    df.loc[(df[Y_OFFSET].isnull()) & (df[VA] == BOTTOM), Y_OFFSET] = default_y_offset

    # Ensure offset not in opposite direction to alignment
    df.loc[(df[HA] == RIGHT) & (df[X_OFFSET] > 0), X_OFFSET] = 0
    df.loc[(df[HA] == LEFT) & (df[X_OFFSET] < 0), X_OFFSET] = 0
    df.loc[(df[VA] == TOP) & (df[Y_OFFSET] > 0), Y_OFFSET] = 0
    df.loc[(df[VA] == BOTTOM) & (df[Y_OFFSET] < 0), Y_OFFSET] = 0

    # Plot the y=x line if selected
    if y_x_line == True:
        ax.axline(
            (0, 0),
            slope=1,
            color=GREY,
            linestyle=DASH_LINE,
            linewidth=2,
            zorder=1,
        )
        ax.text(
            0.99 * x_max, 1.01 * x_max, GOALS_EQUAL_XG, fontsize=20, ha=RIGHT, va=BOTTOM
        )

    # Scale the size of the points if a size metric is provided
    if size_metric:
        sizes = ((df[size_metric]) / (df[size_metric].max())) * 400 + 50
    else:
        sizes = 300

    # Plot the scatter plot including the labels for each point
    for _, row in df.iterrows():
        ax.scatter(
            row[x_metric],
            row[y_metric],
            color=row[COLOR],
            edgecolor=WHITE,
            s=sizes if isinstance(sizes, int) else sizes[row.name],
            linewidth=2,
            zorder=2,
        )
        ax.text(
            row[x_metric] + row[X_OFFSET],
            row[y_metric] + row[Y_OFFSET],
            row[PLAYER_TEAM_COMBO],
            fontsize=20,
            ha=row[HA],
            va=row[VA],
            zorder=3,
        )

    # Set axis limits and labels
    ax.margins(x=0.02, y=0.02)
    ax.tick_params(axis=BOTH, labelsize=20)
    ax.grid(True, linestyle=DASH_LINE, alpha=0.5)
    x_label = wrap_label(CHART_LABELS[x_metric], 85)
    y_label = wrap_label(CHART_LABELS[y_metric], 55)
    ax.set_xlabel(x_label, fontsize=20, labelpad=8)
    ax.set_ylabel(y_label, fontsize=20, labelpad=8)

    # Save the scatter plot as an image file in the data/outputs folder
    fig.tight_layout()
    fig.savefig(
        f"{DATA_OUTPUTS_PATH}{x_metric}_vs_{y_metric}_{SCATTER_PLOT_NAME}",
        bbox_inches=TIGHT,
    )


def create_box_plot(df: pd.DataFrame, y_metric: str, colour_mapping: dict):
    """
    For each player in a dataset, plot box plots for a given metric on the same axis. The box plot is created using seaborn
    and the color of each player's box is determined by the color mapping provided. The box plot is saved as an image file in
    the data/outputs folder.

    Args:
        df (pd.DataFrame): DataFrame containing the metric to be plotted
        y_metric (str): The metric to be plotted on the y-axis
        colour_mapping (dict): A dictionary mapping each player to a color
    """

    # Order the players by median value of the metric to ensure highest median is plotted first
    player_order = (
        df.groupby(PLAYER_NAME)[y_metric].median().sort_values(ascending=False).index
    )

    # Plot box plots for each player on the same axis
    fig, ax = plt.subplots(figsize=(15, 9.5))
    sns.boxplot(
        data=df,
        x=y_metric,
        y=PLAYER_NAME,
        order=player_order,
        palette=[colour_mapping[player] for player in player_order],
    )
    ax.tick_params(axis=BOTH, labelsize=20)
    ax.grid(True, linestyle=DASH_LINE, axis=X, alpha=0.5)

    ax.set_xlabel(CHART_LABELS[y_metric], fontsize=20, labelpad=8)
    ax.set_ylabel(None)

    # Save the box plots as an image file in the data/outputs folder
    fig.tight_layout()
    fig.savefig(
        f"{DATA_OUTPUTS_PATH}{y_metric}_{BOX_PLOT_NAME}",
        bbox_inches=TIGHT,
    )


def create_radar_plot(
    df: pd.DataFrame, player_colour_mapping: dict, chart_labels: dict
):
    """
    Create a radar plot of the metrics in a dataset for each player. The radar plot is created using matplotlib and the color
    of each player's radar plot is determined by the color mapping provided. The radar plot is saved as an image file in the
    data/outputs folder.

    Args:
        df (pd.DataFrame): DataFrame containing the metrics to be plotted
        player_colour_mapping (dict): A dictionary mapping each player to a color
        chart_labels (dict): A dictionary mapping each metric to a label
    """

    # Define the metrics to be plotted on the radar chart
    metrics = [metric for metric in df.columns if metric != PLAYER_NAME]

    # Calculate the number of metrics and there fore angles for the radar plot
    num_metrics = len(metrics)
    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    # Plot the radar chart for each player
    for _, player in df.iterrows():
        values = player[metrics].tolist()
        values += values[:1]
        ax.plot(
            angles,
            values,
            label=player[PLAYER_NAME],
            color=player_colour_mapping[player[PLAYER_NAME]],
            linewidth=2,
        )
        ax.fill(
            angles, values, alpha=0.2, color=player_colour_mapping[player[PLAYER_NAME]]
        )

    # Set the labels for each metric on the radar chart
    labels = [wrap_label(chart_labels[metric], 30) for metric in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, ha=CENTER)

    # Set the alignment of the metric labels on the radar chart
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if (0 <= angle < np.pi / 2) or (3 * np.pi / 2 < angle < 2 * np.pi):
            label.set_ha(LEFT)
            label.set_va(CENTER)
        elif angle == np.pi / 2:
            label.set_ha(CENTER)
            label.set_va(BOTTOM)
        elif np.pi / 2 < angle < (3 * np.pi) / 2:
            label.set_ha(RIGHT)
            label.set_va(CENTER)
        elif angle == (3 * np.pi) / 2:
            label.set_ha(CENTER)
            label.set_va(TOP)

    # Set the y-ticks and grid lines for the radar chart
    ax.set_yticks([50, 75, 100])
    ax.set_yticklabels(RADAR_PLOT_Y_TICK_LABELS, color=BLACK, fontsize=10)
    ax.yaxis.grid(True, color=BLACK, linestyle=DASH_LINE, alpha=0.5)

    # Remove the radial spines from the radar chart
    ax.spines[POLAR].set_visible(False)
    ax.grid(color=BLACK, linestyle=STRAIGHT_LINE, linewidth=1, alpha=0.5)

    # Add a legend to the radar chart
    ax.legend(
        loc=UPPER_RIGHT,
        bbox_to_anchor=(1.5, 1.1),
        title=PLAYER,
        fontsize=12,
        title_fontsize=12,
    )

    # Save the radar plot as an image file in the data/outputs folder
    fig.tight_layout()
    fig.savefig(f"{DATA_OUTPUTS_PATH}{RADAR_PLOT_NAME}", bbox_inches=TIGHT)
