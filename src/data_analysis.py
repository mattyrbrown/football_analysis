import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import seaborn as sns
from typing import Optional
from adjustText import adjust_text
import textwrap
from mplsoccer import Pitch, VerticalPitch
from math import pi
from plotnine import (
    ggplot, aes, geom_polygon, geom_path, geom_line, geom_point, geom_text, labs, theme_void, theme,
    element_blank, element_text, lims
)
from sklearn.preprocessing import MinMaxScaler

from src.config import *


def player_team_dataset():

    # Read in merged events and match data
    events_matches_df = pd.read_csv(
        f"{DATA_PROCESSED_PATH}{MERGED_EVENTS_MATCH_LEVEL_DATA_NAME}"
    )

    events_matches_df = events_matches_df.groupby(
        [PLAYER_ID, PLAYER_NAME, MATCH_ID], as_index=False, dropna=False
    )[[TEAM_ELO, OPPOSITION_ELO, ELO_DIFFERENCE]].first()

    return events_matches_df


def possesions_dataset():

    # Read in processed events data
    events_df = pd.read_csv(
        f"{DATA_PROCESSED_PATH}{MERGED_EVENTS_MATCH_LEVEL_DATA_NAME}"
    )

    # Find ball receipt events
    posessions = events_df[
        (events_df[TYPE_NAME] != PRESSURE)
        & (
            (events_df[DUEL_OUTCOME_NAME] == WON)
            | events_df[DUEL_OUTCOME_NAME].isnull()
        )
        & (events_df[START_OF_POSSESSION] == True)
        & (events_df[PENALTY] == False)
    ]
    posessions = posessions[
        [
            MATCH_ID,
            TEAM_ID,
            TEAM_NAME,
            PLAYER_ID,
            PLAYER_NAME,
            MINUTES_PLAYED,
            POSSESSION_INDEX,
            TYPE_NAME,
            LOCATION_X,
            LOCATION_Y,
            OPPOSITION_ELO_WEIGHTING,
            ELO_DIFFERENCE_WEIGHTING,
        ]
    ]

    # Label receptions that take place in zone 14 and 17
    posessions[ZONE_14_AND_17] = False
    posessions.loc[
        (posessions[LOCATION_X] >= 80)
        & (posessions[LOCATION_Y] >= 20)
        & (posessions[LOCATION_Y] <= 60),
        ZONE_14_AND_17,
    ] = True

    # Label receptions in final third
    posessions[FINAL_THIRD] = False
    posessions.loc[
        posessions[LOCATION_X] >= 80,
        FINAL_THIRD,
    ] = True

    # Find possession indexes associated to ball receipts
    possession_indexes = posessions[POSSESSION_INDEX].unique()
    possessions_events = events_df[events_df[POSSESSION_INDEX].isin(possession_indexes)]

    # Find the outcome of ball receipt possessions
    outcome_of_posessions = possessions_events.groupby(
        POSSESSION_INDEX, as_index=False, dropna=False
    ).tail(1)
    outcome_of_posessions.loc[
        outcome_of_posessions[TYPE_NAME] == CARRY, END_OF_EVENT_LOCATION_X
    ] = outcome_of_posessions.loc[
        outcome_of_posessions[TYPE_NAME] == CARRY, CARRY_END_LOCATION_X
    ]
    outcome_of_posessions.loc[
        outcome_of_posessions[TYPE_NAME] == CARRY, END_OF_EVENT_LOCATION_Y
    ] = outcome_of_posessions.loc[
        outcome_of_posessions[TYPE_NAME] == CARRY, CARRY_END_LOCATION_Y
    ]
    outcome_of_posessions.loc[
        outcome_of_posessions[TYPE_NAME] != CARRY, END_OF_EVENT_LOCATION_X
    ] = outcome_of_posessions.loc[outcome_of_posessions[TYPE_NAME] != CARRY, LOCATION_X]
    outcome_of_posessions.loc[
        outcome_of_posessions[TYPE_NAME] != CARRY, END_OF_EVENT_LOCATION_Y
    ] = outcome_of_posessions.loc[outcome_of_posessions[TYPE_NAME] != CARRY, LOCATION_Y]
    outcome_of_posessions = outcome_of_posessions[
        [
            POSSESSION_INDEX,
            TYPE_NAME,
            END_OF_EVENT_LOCATION_X,
            END_OF_EVENT_LOCATION_Y,
            GOAL,
            PASS_ASSISTED_SHOT_ID,
            EXPECTED_GOALS,
        ]
    ]

    outcome_of_posessions = outcome_of_posessions.rename(
        columns={TYPE_NAME: OUTCOME_TYPE_NAME}
    )

    outcome_of_posessions[SHOOTING_OPPORTUNITY] = False
    outcome_of_posessions.loc[
        (outcome_of_posessions[OUTCOME_TYPE_NAME] == SHOT)
        | (outcome_of_posessions[PASS_ASSISTED_SHOT_ID].notnull()),
        SHOOTING_OPPORTUNITY,
    ] = True

    # Find the total sum of event value for ball receipt possessions
    event_score_possessions = possessions_events.groupby(
        POSSESSION_INDEX, as_index=False, dropna=False
    )[EVENT_VALUE].agg(possession_event_score="sum", events="count")

    # Merge dataframes to create the receptions dataset
    posessions = posessions.merge(
        outcome_of_posessions,
        on=POSSESSION_INDEX,
        how=LEFT,
    )
    posessions = posessions.merge(
        event_score_possessions,
        on=POSSESSION_INDEX,
        how=LEFT,
    )

    # Return the receptions dataset
    return posessions


def plot_receptions_on_pitch(receptions):

    # path effects
    path_eff = [
        path_effects.Stroke(linewidth=3, foreground=BLACK),
        path_effects.Normal(),
    ]

    # Create a pitch
    pitch = VerticalPitch(
        pitch_type=PITCH_TYPE,
        pitch_color=PITCH_COLOUR,
        line_color=PITCH_COLOUR,
        line_zorder=2,
    )

    bin_x = np.linspace(pitch.dim.left, pitch.dim.right, num=7)
    bin_y = np.sort(np.array([pitch.dim.bottom, 20, 60, pitch.dim.top]))

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

    for player_id in PLAYER_IDS:

        df = receptions[receptions[PLAYER_ID] == player_id]

        # Create a figure
        fig, ax = pitch.draw(figsize=(12, 8))

        # Set the facecolor of the figure
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
            weight="bold",
        )

        # plt.show()

        # Save the plot
        fig.savefig(
            f"{DATA_OUTPUTS_PATH}{PLAYER}_{player_id}_{RECEPTIONS_HEATMAP_NAME}",
            bbox_inches=TIGHT,
        )


def calcluate_values_for_receptions(receptions: pd.DataFrame, subset: list):

    if MATCH_ID in subset:
        total_minutes_per_player = receptions.groupby(
            subset, as_index=False, dropna=False
        )[[MINUTES_PLAYED, OPPOSITION_ELO_WEIGHTING, ELO_DIFFERENCE_WEIGHTING]].max()
    else:
        total_minutes_per_player = receptions.groupby(
            subset + [MATCH_ID], as_index=False, dropna=False
        )[MINUTES_PLAYED].max()
        total_minutes_per_player = total_minutes_per_player.groupby(
            subset, as_index=False, dropna=False
        )[MINUTES_PLAYED].sum()

    total_receptions_per_player = receptions.groupby(
        subset, as_index=False, dropna=False
    )[POSSESSION_INDEX].count()
    total_receptions_per_player = total_receptions_per_player.rename(
        columns={POSSESSION_INDEX: TOTAL_RECEPTIONS}
    )

    total_receptions_in_zone_14_and_17_per_player = (
        receptions[receptions[ZONE_14_AND_17] == True]
        .groupby(subset, as_index=False, dropna=False)[POSSESSION_INDEX]
        .count()
    )
    total_receptions_in_zone_14_and_17_per_player = (
        total_receptions_in_zone_14_and_17_per_player.rename(
            columns={POSSESSION_INDEX: TOTAL_RECEPTIONS_IN_ZONE_14_AND_17}
        )
    )

    total_receptions_in_final_third_per_player = (
        receptions[receptions[FINAL_THIRD] == True]
        .groupby(subset, as_index=False, dropna=False)[POSSESSION_INDEX]
        .count()
    )
    total_receptions_in_final_third_per_player = (
        total_receptions_in_final_third_per_player.rename(
            columns={POSSESSION_INDEX: TOTAL_RECEPTIONS_IN_FINAL_THIRD}
        )
    )

    total_shooting_opportunities_in_zone_14_and_17_per_player = (
        receptions[
            (receptions[SHOOTING_OPPORTUNITY] == True)
            & (receptions[ZONE_14_AND_17] == True)
        ]
        .groupby(subset, as_index=False, dropna=False)[POSSESSION_INDEX]
        .count()
    )
    total_shooting_opportunities_in_zone_14_and_17_per_player = (
        total_shooting_opportunities_in_zone_14_and_17_per_player.rename(
            columns={POSSESSION_INDEX: TOTAL_SHOOTING_OPPORTUNITIES_IN_ZONE_14_AND_17}
        )
    )

    shots_in_zone_14_and_17_per_player = (
        receptions[
            (receptions[OUTCOME_TYPE_NAME] == SHOT)
            & (receptions[ZONE_14_AND_17] == True)
        ]
        .groupby(subset, as_index=False, dropna=False)[EXPECTED_GOALS]
        .agg(
            total_shots_in_zone_14_and_17=COUNT,
            total_expected_goals_in_zone_14_and_17=SUM,
            mean_expected_goals_per_shot_from_receptions_in_zone_14_and_17=MEAN,
            median_expected_goals_per_shot_from_receptions_in_zone_14_and_17=MEDIAN,
        )
    )

    average_event_score_per_reception_in_zone_14_and_17_per_player = (
        receptions[receptions[ZONE_14_AND_17] == True]
        .groupby(subset, as_index=False, dropna=False)[POSSESSION_EVENT_SCORE]
        .agg(
            mean_event_score_per_reception_in_zone_14_and_17=MEAN,
            median_event_score_per_reception_in_zone_14_and_17=MEDIAN,
        )
    )

    player_reception_metrics = total_minutes_per_player.merge(
        total_receptions_per_player, on=subset, how=LEFT
    )
    player_reception_metrics = player_reception_metrics.merge(
        total_receptions_in_zone_14_and_17_per_player,
        on=subset,
        how=LEFT,
    )
    player_reception_metrics = player_reception_metrics.merge(
        total_receptions_in_final_third_per_player,
        on=subset,
        how=LEFT,
    )
    player_reception_metrics = player_reception_metrics.merge(
        total_shooting_opportunities_in_zone_14_and_17_per_player,
        on=subset,
        how=LEFT,
    )
    player_reception_metrics = player_reception_metrics.merge(
        shots_in_zone_14_and_17_per_player, on=subset, how=LEFT
    )
    player_reception_metrics = player_reception_metrics.merge(
        average_event_score_per_reception_in_zone_14_and_17_per_player,
        on=subset,
        how=LEFT,
    )

    return player_reception_metrics


def calculate_reception_metrics(
    player_reception_metrics: pd.DataFrame, subset: list[str]
):

    # Determine metrics per 90 or per reception
    player_reception_metrics[RECEPTIONS_PER_90] = (
        player_reception_metrics[TOTAL_RECEPTIONS]
        / player_reception_metrics[MINUTES_PLAYED]
        * 90
    )
    player_reception_metrics[RECEPTIONS_IN_ZONE_14_AND_17_PER_90] = (
        player_reception_metrics[TOTAL_RECEPTIONS_IN_ZONE_14_AND_17]
        / player_reception_metrics[MINUTES_PLAYED]
        * 90
    )
    player_reception_metrics[PERCENTAGE_RECEPTIONS_IN_ZONE_14_AND_17] = (
        player_reception_metrics[TOTAL_RECEPTIONS_IN_ZONE_14_AND_17]
        / player_reception_metrics[TOTAL_RECEPTIONS]
        * 100
    )
    player_reception_metrics[RECEPTIONS_IN_FINAL_THIRD_PER_90] = (
        player_reception_metrics[TOTAL_RECEPTIONS_IN_FINAL_THIRD]
        / player_reception_metrics[MINUTES_PLAYED]
        * 90
    )
    player_reception_metrics[PERCENTAGE_RECEPTIONS_IN_FINAL_THIRD] = (
        player_reception_metrics[TOTAL_RECEPTIONS_IN_FINAL_THIRD]
        / player_reception_metrics[TOTAL_RECEPTIONS]
        * 100
    )
    player_reception_metrics[SHOOTING_OPPORTUNITIES_PER_RECEPTION_IN_ZONE_14_AND_17] = (
        player_reception_metrics[TOTAL_SHOOTING_OPPORTUNITIES_IN_ZONE_14_AND_17]
        / player_reception_metrics[TOTAL_RECEPTIONS_IN_ZONE_14_AND_17]
    )
    player_reception_metrics[
        SHOOTING_OPPORTUNITIES_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17
    ] = (
        player_reception_metrics[TOTAL_SHOOTING_OPPORTUNITIES_IN_ZONE_14_AND_17]
        / player_reception_metrics[MINUTES_PLAYED]
        * 90
    )
    player_reception_metrics[SHOTS_PER_RECEPTION_IN_ZONE_14_AND_17] = (
        player_reception_metrics[TOTAL_SHOTS_IN_ZONE_14_AND_17]
        / player_reception_metrics[TOTAL_RECEPTIONS_IN_ZONE_14_AND_17]
    )
    player_reception_metrics[SHOTS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17] = (
        player_reception_metrics[TOTAL_SHOTS_IN_ZONE_14_AND_17]
        / player_reception_metrics[MINUTES_PLAYED]
        * 90
    )
    player_reception_metrics[
        EXPECTED_GOALS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17
    ] = (
        player_reception_metrics[TOTAL_EXPECTED_GOALS_IN_ZONE_14_AND_17]
        / player_reception_metrics[MINUTES_PLAYED]
        * 90
    )

    if MATCH_ID in subset:
        player_reception_metrics = player_reception_metrics[
            subset
            + [
                MINUTES_PLAYED,
                OPPOSITION_ELO_WEIGHTING,
                ELO_DIFFERENCE_WEIGHTING,
                RECEPTIONS_PER_90,
                RECEPTIONS_IN_ZONE_14_AND_17_PER_90,
                RECEPTIONS_IN_FINAL_THIRD_PER_90,
                SHOOTING_OPPORTUNITIES_PER_RECEPTION_IN_ZONE_14_AND_17,
                SHOOTING_OPPORTUNITIES_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17,
                SHOTS_PER_RECEPTION_IN_ZONE_14_AND_17,
                SHOTS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17,
                EXPECTED_GOALS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17,
                MEAN_EXPECTED_GOALS_PER_SHOT_FROM_RECEPTIONS_IN_ZONE_14_AND_17,
                MEDIAN_EXPECTED_GOALS_PER_SHOT_FROM_RECEPTIONS_IN_ZONE_14_AND_17,
                MEAN_EVENT_SCORE_PER_RECEPTION_IN_ZONE_14_AND_17,
                MEDIAN_EVENT_SCORE_PER_RECEPTION_IN_ZONE_14_AND_17,
            ]
        ]
    else:
        player_reception_metrics = player_reception_metrics[
            subset
            + [
                RECEPTIONS_PER_90,
                RECEPTIONS_IN_ZONE_14_AND_17_PER_90,
                PERCENTAGE_RECEPTIONS_IN_ZONE_14_AND_17,
                RECEPTIONS_IN_FINAL_THIRD_PER_90,
                PERCENTAGE_RECEPTIONS_IN_FINAL_THIRD,
                SHOOTING_OPPORTUNITIES_PER_RECEPTION_IN_ZONE_14_AND_17,
                SHOOTING_OPPORTUNITIES_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17,
                SHOTS_PER_RECEPTION_IN_ZONE_14_AND_17,
                SHOTS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17,
                EXPECTED_GOALS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17,
                MEAN_EXPECTED_GOALS_PER_SHOT_FROM_RECEPTIONS_IN_ZONE_14_AND_17,
                MEDIAN_EXPECTED_GOALS_PER_SHOT_FROM_RECEPTIONS_IN_ZONE_14_AND_17,
                MEAN_EVENT_SCORE_PER_RECEPTION_IN_ZONE_14_AND_17,
                MEDIAN_EVENT_SCORE_PER_RECEPTION_IN_ZONE_14_AND_17,
            ]
        ]

    return player_reception_metrics


def normalise_metrics(df: pd.DataFrame, metric_columns: list):

    for column in metric_columns:
        df[column] = (
            df[column] * df[OPPOSITION_ELO_WEIGHTING] * df[ELO_DIFFERENCE_WEIGHTING]
        )
        df[column] = df[column] * df[MINUTES_PLAYED]

    df = df.groupby([PLAYER_ID, PLAYER_NAME], as_index=False, dropna=False)[
        metric_columns + [MINUTES_PLAYED]
    ].sum()

    for column in metric_columns:
        df[column] = df[column] / df[MINUTES_PLAYED]
        df[column] = (df[column] * 100) / df[column].max()

    return df


def output_normalised_metrics_for_receptions(receptions: pd.DataFrame):

    player_normalised_reception_metrics = calcluate_values_for_receptions(
        receptions, [MATCH_ID, PLAYER_ID, PLAYER_NAME]
    )
    player_normalised_reception_metrics = calculate_reception_metrics(
        player_normalised_reception_metrics, [MATCH_ID, PLAYER_ID, PLAYER_NAME]
    )

    metric_columns = [
        column
        for column in player_normalised_reception_metrics.columns
        if column
        not in [
            MATCH_ID,
            PLAYER_ID,
            PLAYER_NAME,
            MINUTES_PLAYED,
            OPPOSITION_ELO_WEIGHTING,
            ELO_DIFFERENCE_WEIGHTING,
        ]
    ]

    player_normalised_reception_metrics = normalise_metrics(
        player_normalised_reception_metrics, metric_columns
    )

    player_normalised_reception_metrics.to_csv(
        f"{DATA_OUTPUTS_PATH}{PLAYER}_{RECEPTIONS_NORMALISED_METRICS_NAME}",
        index=False,
    )

    return player_normalised_reception_metrics


def output_metrics_for_receptions(
    receptions: pd.DataFrame, subset=[PLAYER_ID, PLAYER_NAME]
):

    player_reception_metrics = calcluate_values_for_receptions(receptions, subset)
    player_reception_metrics = calculate_reception_metrics(
        player_reception_metrics, subset
    )

    # Save the reception metrics
    if subset == [PLAYER_ID, PLAYER_NAME]:
        player_reception_metrics.to_csv(
            f"{DATA_OUTPUTS_PATH}{PLAYER}_{RECEPTIONS_METRICS_NAME}",
            index=False,
        )
    elif subset == [PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_NAME]:
        player_reception_metrics.to_csv(
            f"{DATA_OUTPUTS_PATH}{PLAYER}_{TEAM}_{RECEPTIONS_METRICS_NAME}",
            index=False,
        )

    return player_reception_metrics


def calcluate_values_for_goals(posessions: pd.DataFrame, subset):

    if MATCH_ID in subset:
        total_minutes_per_player = posessions.groupby(
            subset, as_index=False, dropna=False
        )[[MINUTES_PLAYED, OPPOSITION_ELO_WEIGHTING, ELO_DIFFERENCE_WEIGHTING]].max()
    else:
        total_minutes_per_player = posessions.groupby(
            subset + [MATCH_ID], as_index=False, dropna=False
        )[MINUTES_PLAYED].max()
        total_minutes_per_player = total_minutes_per_player.groupby(
            subset, as_index=False, dropna=False
        )[MINUTES_PLAYED].sum()

    posessions_per_player = posessions.groupby(subset, as_index=False, dropna=False)[
        POSSESSION_INDEX
    ].count()
    posessions_per_player = posessions_per_player.rename(
        columns={POSSESSION_INDEX: TOTAL_POSSESSIONS}
    )

    total_shots_per_player = (
        posessions[posessions[OUTCOME_TYPE_NAME] == SHOT]
        .groupby(subset, as_index=False, dropna=False)[POSSESSION_INDEX]
        .count()
    )
    total_shots_per_player = total_shots_per_player.rename(
        columns={POSSESSION_INDEX: TOTAL_SHOTS}
    )

    total_shooting_opportunities_per_player = (
        posessions[posessions[SHOOTING_OPPORTUNITY] == True]
        .groupby(subset, as_index=False, dropna=False)[POSSESSION_INDEX]
        .count()
    )
    total_shooting_opportunities_per_player = (
        total_shooting_opportunities_per_player.rename(
            columns={POSSESSION_INDEX: TOTAL_SHOOTING_OPPORTUNITIES}
        )
    )

    total_goals_per_player = (
        posessions[posessions[GOAL] == True]
        .groupby(subset, as_index=False, dropna=False)[POSSESSION_INDEX]
        .count()
    )
    total_goals_per_player = total_goals_per_player.rename(
        columns={POSSESSION_INDEX: TOTAL_GOALS}
    )

    expected_goals_per_player = (
        posessions[posessions[OUTCOME_TYPE_NAME] == SHOT]
        .groupby(subset, as_index=False, dropna=False)[EXPECTED_GOALS]
        .agg(
            total_expected_goals=SUM,
            mean_expected_goals_per_shot=MEAN,
            median_expected_goals_per_shot=MEDIAN,
        )
    )

    # Merge player-level values for goals
    player_goal_metrics = total_minutes_per_player.merge(
        posessions_per_player, on=subset, how=LEFT
    )
    player_goal_metrics = player_goal_metrics.merge(
        total_shots_per_player, on=subset, how=LEFT
    )
    player_goal_metrics = player_goal_metrics.merge(
        total_shooting_opportunities_per_player, on=subset, how=LEFT
    )
    player_goal_metrics = player_goal_metrics.merge(
        total_goals_per_player, on=subset, how=LEFT
    )
    player_goal_metrics = player_goal_metrics.merge(
        expected_goals_per_player, on=subset, how=LEFT
    )

    return player_goal_metrics


def calculate_goals_metrics(player_goal_metrics: pd.DataFrame, subset: list[str]):

    # Determine metrics per 90 or per posession
    player_goal_metrics[SHOOTING_OPPORTUNITIES_PER_90] = (
        player_goal_metrics[TOTAL_SHOOTING_OPPORTUNITIES]
        / player_goal_metrics[MINUTES_PLAYED]
        * 90
    )
    player_goal_metrics[SHOOTING_OPPORTUNITIES_PER_POSSESSION] = (
        player_goal_metrics[TOTAL_SHOOTING_OPPORTUNITIES]
        / player_goal_metrics[TOTAL_POSSESSIONS]
    )
    player_goal_metrics[SHOTS_PER_90] = (
        player_goal_metrics[TOTAL_SHOTS] / player_goal_metrics[MINUTES_PLAYED] * 90
    )
    player_goal_metrics[SHOTS_PER_POSSESSION] = (
        player_goal_metrics[TOTAL_SHOTS] / player_goal_metrics[TOTAL_POSSESSIONS]
    )
    player_goal_metrics[GOALS_PER_90] = (
        player_goal_metrics[TOTAL_GOALS] / player_goal_metrics[MINUTES_PLAYED] * 90
    )
    player_goal_metrics[GOALS_PER_SHOT] = (
        player_goal_metrics[TOTAL_GOALS] / player_goal_metrics[TOTAL_SHOTS]
    )
    player_goal_metrics[GOALS_PER_POSSESSION] = (
        player_goal_metrics[TOTAL_GOALS] / player_goal_metrics[TOTAL_POSSESSIONS]
    )
    player_goal_metrics[EXPECTED_GOALS_PER_90] = (
        player_goal_metrics[TOTAL_EXPECTED_GOALS]
        / player_goal_metrics[MINUTES_PLAYED]
        * 90
    )
    player_goal_metrics[RATIO_OF_GOALS_TO_EXPECTED_GOALS] = (
        player_goal_metrics[TOTAL_GOALS] / player_goal_metrics[TOTAL_EXPECTED_GOALS]
    )

    if MATCH_ID in subset:
        player_goal_metrics = player_goal_metrics[
            subset
            + [
                MINUTES_PLAYED,
                OPPOSITION_ELO_WEIGHTING,
                ELO_DIFFERENCE_WEIGHTING,
                SHOOTING_OPPORTUNITIES_PER_90,
                SHOOTING_OPPORTUNITIES_PER_POSSESSION,
                SHOTS_PER_90,
                SHOTS_PER_POSSESSION,
                GOALS_PER_90,
                GOALS_PER_SHOT,
                GOALS_PER_POSSESSION,
                EXPECTED_GOALS_PER_90,
                MEAN_EXPECTED_GOALS_PER_SHOT,
                MEDIAN_EXPECTED_GOALS_PER_SHOT,
                RATIO_OF_GOALS_TO_EXPECTED_GOALS,
            ]
        ]
    else:
        player_goal_metrics = player_goal_metrics[
            subset
            + [
                SHOOTING_OPPORTUNITIES_PER_90,
                SHOOTING_OPPORTUNITIES_PER_POSSESSION,
                SHOTS_PER_90,
                SHOTS_PER_POSSESSION,
                GOALS_PER_90,
                GOALS_PER_SHOT,
                GOALS_PER_POSSESSION,
                EXPECTED_GOALS_PER_90,
                MEAN_EXPECTED_GOALS_PER_SHOT,
                MEDIAN_EXPECTED_GOALS_PER_SHOT,
                RATIO_OF_GOALS_TO_EXPECTED_GOALS,
            ]
        ]

    return player_goal_metrics


def output_normalised_metrics_for_goals(possessions: pd.DataFrame):

    player_normalised_goals_metrics = calcluate_values_for_goals(
        possessions, [MATCH_ID, PLAYER_ID, PLAYER_NAME]
    )
    player_normalised_goals_metrics = calculate_goals_metrics(
        player_normalised_goals_metrics, [MATCH_ID, PLAYER_ID, PLAYER_NAME]
    )

    metric_columns = [
        column
        for column in player_normalised_goals_metrics.columns
        if column
        not in [
            MATCH_ID,
            PLAYER_ID,
            PLAYER_NAME,
            MINUTES_PLAYED,
            OPPOSITION_ELO_WEIGHTING,
            ELO_DIFFERENCE_WEIGHTING,
        ]
    ]

    player_normalised_goals_metrics = normalise_metrics(
        player_normalised_goals_metrics, metric_columns
    )

    player_normalised_goals_metrics.to_csv(
        f"{DATA_OUTPUTS_PATH}{PLAYER}_{GOALS_NORMALISED_METRICS_NAME}",
        index=False,
    )

    return player_normalised_goals_metrics


def output_metrics_for_goals(posessions: pd.DataFrame, subset=[PLAYER_ID, PLAYER_NAME]):

    player_goal_metrics = calcluate_values_for_goals(posessions, subset)
    player_goal_metrics = calculate_goals_metrics(player_goal_metrics, subset)

    # Save the goal metrics
    if subset == [PLAYER_ID, PLAYER_NAME]:
        player_goal_metrics.to_csv(
            f"{DATA_OUTPUTS_PATH}{PLAYER}_{GOAL_METRICS_NAME}",
            index=False,
        )
    elif subset == [PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_NAME]:
        player_goal_metrics.to_csv(
            f"{DATA_OUTPUTS_PATH}{PLAYER}_{TEAM}_{GOAL_METRICS_NAME}",
            index=False,
        )

    return player_goal_metrics


def create_receptions_slide_table(
    player_receptions_metrics, normalised_player_receptions_metrics
):

    # Select required metrics
    receptions_slide_table = player_receptions_metrics[
        [
            PLAYER_NAME,
            RECEPTIONS_IN_ZONE_14_AND_17_PER_90,
            PERCENTAGE_RECEPTIONS_IN_ZONE_14_AND_17,
            PERCENTAGE_RECEPTIONS_IN_FINAL_THIRD,
        ]
    ]

    receptions_slide_table = receptions_slide_table.merge(
        normalised_player_receptions_metrics[
            [
                PLAYER_NAME,
                RECEPTIONS_IN_ZONE_14_AND_17_PER_90,
            ]
        ].rename(
            columns={
                RECEPTIONS_IN_ZONE_14_AND_17_PER_90: f"{NORMALISED}_{RECEPTIONS_IN_ZONE_14_AND_17_PER_90}"
            }
        ),
        on=PLAYER_NAME,
        how=LEFT,
    )

    receptions_slide_table = receptions_slide_table[
        [
            PLAYER_NAME,
            RECEPTIONS_IN_ZONE_14_AND_17_PER_90,
            f"{NORMALISED}_{RECEPTIONS_IN_ZONE_14_AND_17_PER_90}",
            PERCENTAGE_RECEPTIONS_IN_ZONE_14_AND_17,
            PERCENTAGE_RECEPTIONS_IN_FINAL_THIRD,
        ]
    ]

    # Flip columns and rows
    receptions_slide_table = receptions_slide_table.transpose()
    receptions_slide_table.columns = receptions_slide_table.iloc[0]
    receptions_slide_table = receptions_slide_table[1:]

    # Save receptions slide table
    receptions_slide_table.to_csv(
        f"{DATA_OUTPUTS_PATH}{RECEPTIONS_SLIDE_TABLE_NAME}",
        index=True,
    )


def create_radar_plot_inputs(
    normalised_player_receptions_metrics,
    player_goals_metrics,
    normalised_player_goals_metrics,
):

    normalised_player_receptions_metrics = normalised_player_receptions_metrics[
        [
            PLAYER_NAME,
            SHOOTING_OPPORTUNITIES_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17,
            EXPECTED_GOALS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17,
        ]
    ].rename(
        columns={
            SHOOTING_OPPORTUNITIES_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17: f"{NORMALISED}_{SHOOTING_OPPORTUNITIES_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17}",
            EXPECTED_GOALS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17: f"{NORMALISED}_{EXPECTED_GOALS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17}",
        }
    )
    normalised_player_goals_metrics = normalised_player_goals_metrics[
        [PLAYER_NAME, GOALS_PER_90]
    ].rename(columns={GOALS_PER_90: f"{NORMALISED}_{GOALS_PER_90}"})
    player_goals_metrics = player_goals_metrics[
        [
            PLAYER_NAME,
            SHOOTING_OPPORTUNITIES_PER_POSSESSION,
            RATIO_OF_GOALS_TO_EXPECTED_GOALS,
        ]
    ]

    metrics_to_be_scaled = [
        metric for metric in player_goals_metrics.columns if metric not in [PLAYER_NAME]
    ]

    for metric in metrics_to_be_scaled:
        player_goals_metrics[metric] = (
            player_goals_metrics[metric] * 100 / player_goals_metrics[metric].max()
        )

    radar_plot_inputs = normalised_player_receptions_metrics.merge(
        normalised_player_goals_metrics, on=PLAYER_NAME, how=LEFT
    )
    radar_plot_inputs = radar_plot_inputs.merge(
        player_goals_metrics, on=PLAYER_NAME, how=LEFT
    )

    return radar_plot_inputs


def closest_difference(df, column):
    values = df[column].values  # Extract values as a numpy array

    closest_diff = []
    for i, val in enumerate(values):
        # Calculate absolute differences and exclude current row (set diff to inf for itself)
        diffs = values - val
        diffs[i] = np.inf  # Exclude the current row
        closest_diff.append(np.min(diffs))  # Minimum of the remaining differences

    return closest_diff


def wrap_label(label, width):
    return "\n".join(textwrap.wrap(label, width))


def create_scatter_plot_of_two_metrics(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    colour_mapping: dict,
    size_metric: Optional[str] = None,
    y_x_line: bool = False,
):

    if TEAM_ID in df.columns:
        df[PLAYER_TEAM_COMBO] = df[PLAYER_NAME] + " (" + df[TEAM_NAME] + ")"
    else:
        df[PLAYER_TEAM_COMBO] = df[PLAYER_NAME]

    # Map colors to each point in the DataFrame
    df[COLOR] = df[PLAYER_NAME].map(colour_mapping)

    # Create a figure
    fig, ax = plt.subplots(figsize=(15, 9.5))

    # Determine axis limits for boundary checks
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

    df[X_DIFF] = closest_difference(df, x_metric)
    df[Y_DIFF] = closest_difference(df, y_metric)

    df.loc[df[x_metric] > x_lim_max, HA] = RIGHT
    df.loc[df[x_metric] < x_lim_min, HA] = LEFT
    df.loc[df[y_metric] > y_lim_max, VA] = TOP
    df.loc[df[y_metric] < y_lim_min, VA] = BOTTOM

    df[X_OFFSET] = None
    df[Y_OFFSET] = None

    overlapping_points = df[
        (np.abs(df[X_DIFF]) <= 2 * x_word_length)
        & (np.abs(df[Y_DIFF]) <= 2 * y_word_height)
    ]
    non_overlapping_points = df[~df.index.isin(overlapping_points.index)]

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

    df = pd.concat([non_overlapping_points, overlapping_points], ignore_index=True)

    df.loc[df[HA].isnull(), HA] = RIGHT
    df.loc[df[VA].isnull(), VA] = TOP
    df.loc[(df[X_OFFSET].isnull()) & (df[HA] == LEFT), X_OFFSET] = default_x_offset
    df.loc[(df[X_OFFSET].isnull()) & (df[HA] == RIGHT), X_OFFSET] = 0 - default_x_offset
    df.loc[(df[Y_OFFSET].isnull()) & (df[VA] == TOP), Y_OFFSET] = 0 - default_y_offset
    df.loc[(df[Y_OFFSET].isnull()) & (df[VA] == BOTTOM), Y_OFFSET] = default_y_offset

    df.loc[(df[HA] == RIGHT) & (df[X_OFFSET] > 0), X_OFFSET] = df.loc[
        (df[HA] == RIGHT) & (df[X_OFFSET] > 0)
    ]
    df.loc[(df[HA] == LEFT) & (df[X_OFFSET] < 0), X_OFFSET] = 0
    df.loc[(df[VA] == TOP) & (df[Y_OFFSET] > 0), Y_OFFSET] = 0
    df.loc[(df[VA] == BOTTOM) & (df[Y_OFFSET] < 0), Y_OFFSET] = 0

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

    # Determine the size of each point
    if size_metric:
        sizes = (
            (df[size_metric]) / (df[size_metric].max())
        ) * 400 + 50  # Scale size to range [50, 250]
    else:
        sizes = 300

    # Plot the scatter plot with colors
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

        # Add text with offset
        ax.text(
            row[x_metric] + row[X_OFFSET],
            row[y_metric] + row[Y_OFFSET],
            row[PLAYER_TEAM_COMBO],
            fontsize=20,
            ha=row[HA],
            va=row[VA],
            zorder=3,
        )

    ax.margins(x=0.02, y=0.02)
    ax.tick_params(axis=BOTH, labelsize=20)
    ax.grid(True, linestyle=DASH_LINE, alpha=0.5)

    # Set axis labels with wrapped text
    x_label = wrap_label(CHART_LABELS[x_metric], 85)
    y_label = wrap_label(CHART_LABELS[y_metric], 55)

    ax.set_xlabel(x_label, fontsize=20, labelpad=8)
    ax.set_ylabel(y_label, fontsize=20, labelpad=8)

    # Save the scatter plot
    fig.tight_layout()
    fig.savefig(
        f"{DATA_OUTPUTS_PATH}{x_metric}_vs_{y_metric}_{SCATTER_PLOT_NAME}",
        bbox_inches=TIGHT,
    )


def create_density_plot(df: pd.DataFrame, x_metric: str, colour_mapping: dict):

    # Create a color palette based on the unique combinations
    unique_players = df[PLAYER_NAME].unique()
    palette = sns.color_palette("husl", len(unique_players))

    # Create a figure
    fig, ax = plt.subplots(figsize=(15, 5.8))

    # Use seaborn kdeplot for each player
    for player in unique_players:
        sns.kdeplot(
            data=df[df[PLAYER_NAME] == player],
            x=x_metric,
            label=player,
            palette=colour_mapping[player],
            fill=True,
            alpha=0.2,
        )

    ax.tick_params(axis=BOTH, labelsize=16)
    ax.grid(True, linestyle=DASH_LINE, alpha=0.5)

    ax.set_xlabel(CHART_LABELS[x_metric], fontsize=16, labelpad=8)
    ax.set_ylabel(DENSITY, fontsize=16, labelpad=8)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title=PLAYER)

    # Save the scatter plot
    fig.tight_layout()
    plt.show()
    # fig.savefig(
    #     f"{DATA_OUTPUTS_PATH}{x_metric}_{DENSITY_PLOT_NAME}",
    #     bbox_inches=TIGHT,
    # )


def create_box_plot(df: pd.DataFrame, y_metric: str, colour_mapping: dict):

    player_order = (
        df.groupby(PLAYER_NAME)[y_metric].median().sort_values(ascending=False).index
    )

    # Create a figure
    fig, ax = plt.subplots(figsize=(15, 9.5))

    # Create a box plot using seaborn
    sns.boxplot(
        data=df,
        x=y_metric,
        y=PLAYER_NAME,
        order=player_order,
        palette=[colour_mapping[player] for player in player_order],
    )

    # Customize the plot
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, linestyle=DASH_LINE, axis="x", alpha=0.5)

    ax.set_xlabel(CHART_LABELS[y_metric], fontsize=20, labelpad=8)
    ax.set_ylabel(None)

    # Save the box plot
    fig.tight_layout()
    fig.savefig(
        f"{DATA_OUTPUTS_PATH}{y_metric}_{BOX_PLOT_NAME}",
        bbox_inches=TIGHT,
    )


def create_radar_plot(df: pd.DataFrame, player_colour_mapping: dict, chart_labels: dict):
    # Define the metrics to be plotted
    metrics = [metric for metric in df.columns if metric != PLAYER_NAME]  # Replace PLAYER_NAME with 'PLAYER_NAME' string if undefined

    # Define the number of variables
    num_metrics = len(metrics)

    # Angles for radar plot
    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]  # Close the radar chart

    # Initialize radar plot with adjusted width
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    # Plot each player's data with assigned colors
    for _, player in df.iterrows():
        values = player[metrics].tolist()
        values += values[:1]  # Close the radar chart
        ax.plot(angles, values, label=player[PLAYER_NAME], 
                color=player_colour_mapping[player[PLAYER_NAME]], linewidth=2)
        ax.fill(angles, values, alpha=0.2, color=player_colour_mapping[player[PLAYER_NAME]])

    # Add metric labels
    labels = [wrap_label(chart_labels[metric], 30) for metric in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, ha='center')

    # Adjust label positions dynamically to avoid overlap
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if (0 <= angle < np.pi/2) or (3*np.pi/2 < angle < 2*np.pi):
            label.set_ha('left')
            label.set_va('center')
        elif angle == np.pi/2:
            label.set_ha('center')
            label.set_va('bottom')
        elif np.pi/2 < angle < (3*np.pi)/2:
            label.set_ha('right')
            label.set_va('center')
        elif angle == (3*np.pi)/2:
            label.set_ha('center')
            label.set_va('top')

    # Add faint y-axis grid lines
    ax.set_yticks([50, 75, 100])
    ax.set_yticklabels(['50', '75', '100'], color='black', fontsize=10)
    ax.yaxis.grid(True, color='black', linestyle='--', alpha=0.5)

    # Set straight edges for the radar chart
    ax.spines['polar'].set_visible(False)
    ax.grid(color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Add legend with title
    ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1.1), title=PLAYER, fontsize=12, title_fontsize=12)

    # Save the radar plot
    fig.tight_layout()
    fig.savefig(f"{DATA_OUTPUTS_PATH}{RADAR_PLOT_NAME}", bbox_inches='tight')


def create_box_plots_for_player_team_metrics(player_team_metrics, colour_mapping):

    create_box_plot(player_team_metrics, OPPOSITION_ELO, colour_mapping)
    create_box_plot(player_team_metrics, ELO_DIFFERENCE, colour_mapping)


def create_scatter_plot_for_reception_metrics(
    player_receptions_metrics, normalised_player_receptions_metrics, colour_mapping
):

    create_scatter_plot_of_two_metrics(
        player_receptions_metrics,
        SHOOTING_OPPORTUNITIES_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17,
        EXPECTED_GOALS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17,
        colour_mapping,
    )

    normalised_player_receptions_metrics = normalised_player_receptions_metrics.rename(
        columns={
            SHOOTING_OPPORTUNITIES_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17: f"{NORMALISED}_{SHOOTING_OPPORTUNITIES_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17}",
            EXPECTED_GOALS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17: f"{NORMALISED}_{EXPECTED_GOALS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17}",
        }
    )

    create_scatter_plot_of_two_metrics(
        normalised_player_receptions_metrics,
        f"{NORMALISED}_{SHOOTING_OPPORTUNITIES_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17}",
        f"{NORMALISED}_{EXPECTED_GOALS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17}",
        colour_mapping,
    )


def create_scatter_plot_for_goal_scoring_metrics(
    player_goals_metrics, normalised_player_goals_metrics, colour_mapping
):

    create_scatter_plot_of_two_metrics(
        player_goals_metrics,
        EXPECTED_GOALS_PER_90,
        GOALS_PER_90,
        colour_mapping,
        y_x_line=True,
    )

    normalised_player_goals_metrics = normalised_player_goals_metrics.rename(
        columns={
            EXPECTED_GOALS_PER_90: f"{NORMALISED}_{EXPECTED_GOALS_PER_90}",
            GOALS_PER_90: f"{NORMALISED}_{GOALS_PER_90}",
        }
    )

    create_scatter_plot_of_two_metrics(
        normalised_player_goals_metrics,
        f"{NORMALISED}_{EXPECTED_GOALS_PER_90}",
        f"{NORMALISED}_{GOALS_PER_90}",
        colour_mapping,
    )


def get_colour_palette_for_players(df):

    unique_players = df[PLAYER_NAME].unique()
    palette = sns.color_palette("husl", len(unique_players))
    player_color_mapping = {
        player: color for player, color in zip(unique_players, palette)
    }

    return player_color_mapping


def analysis():

    # Get the receptions and player team datasets
    possessions = possesions_dataset()
    player_team_metrics = player_team_dataset()

    # Set colour palette for each player
    player_color_mapping = get_colour_palette_for_players(player_team_metrics)

    # Create density plots for player team metrics
    create_box_plots_for_player_team_metrics(player_team_metrics, player_color_mapping)

    receptions = possessions[possessions[TYPE_NAME] == BALL_RECEIPT]

    # Loop through each plaer and create heatmap of receptions
    plot_receptions_on_pitch(receptions)

    # Create output metrics for receptions
    player_receptions_metrics = output_metrics_for_receptions(
        receptions, [PLAYER_ID, PLAYER_NAME]
    )
    player_team_receptions_metrics = output_metrics_for_receptions(
        receptions, [PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_NAME]
    )
    normalised_player_receptions_metrics = output_normalised_metrics_for_receptions(
        receptions
    )

    # Create table for receptions slide
    create_receptions_slide_table(
        player_receptions_metrics, normalised_player_receptions_metrics
    )

    # Create scatter plot for receptions metrics
    create_scatter_plot_for_reception_metrics(
        player_receptions_metrics,
        normalised_player_receptions_metrics,
        player_color_mapping,
    )

    # Create output metrics for goals
    player_goals_metrics = output_metrics_for_goals(
        possessions, [PLAYER_ID, PLAYER_NAME]
    )
    player_team_goals_metrics = output_metrics_for_goals(
        possessions, [PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_NAME]
    )
    normalised_player_goals_metrics = output_normalised_metrics_for_goals(possessions)

    # Create scatter plots for goal scoring metrics
    create_scatter_plot_for_goal_scoring_metrics(
        player_goals_metrics, normalised_player_goals_metrics, player_color_mapping
    )

    # Create radar plot of key metrics
    radar_plot_inputs = create_radar_plot_inputs(
        normalised_player_receptions_metrics,
        player_goals_metrics,
        normalised_player_goals_metrics,
    )

    # Create radar plot
    create_radar_plot(radar_plot_inputs, player_color_mapping, CHART_LABELS)

if __name__ == "__main__":
    analysis()
