import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch

from src.config import *


def possesions_dataset():

    # Read in processed events data
    events_df = pd.read_csv(f"{DATA_PROCESSED_PATH}{TRANSFOMED_EVENTS_DATA_NAME}")

    # Find ball receipt events
    posessions = events_df[(events_df[TYPE_NAME] != PRESSURE) & ((events_df[DUEL_OUTCOME_NAME] == WON) | events_df[DUEL_OUTCOME_NAME].isnull()) & (events_df[START_OF_POSSESSION] == True)]
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
    possessions_events = events_df[
        events_df[POSSESSION_INDEX].isin(possession_indexes)
    ]

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
    ] = outcome_of_posessions.loc[
        outcome_of_posessions[TYPE_NAME] != CARRY, LOCATION_X
    ]
    outcome_of_posessions.loc[
        outcome_of_posessions[TYPE_NAME] != CARRY, END_OF_EVENT_LOCATION_Y
    ] = outcome_of_posessions.loc[
        outcome_of_posessions[TYPE_NAME] != CARRY, LOCATION_Y
    ]
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
    pitch = Pitch(
        pitch_type=PITCH_TYPE,
        pitch_color=PITCH_COLOUR,
        line_color=PITCH_COLOUR,
        line_zorder=2,
    )

    bin_x = np.linspace(pitch.dim.left, pitch.dim.right, num=7)
    bin_y = np.sort(np.array([pitch.dim.bottom, 20, 60, pitch.dim.top]))

    global_max = 0
    for player_id in range(5):
        df = receptions[receptions[PLAYER_ID] == player_id]
        bin_statistic = pitch.bin_statistic(
            df[LOCATION_X],
            df[LOCATION_Y],
            statistic=COUNT,
            bins=(bin_x, bin_y),
            normalize=True,
        )
        global_max = max(global_max, bin_statistic[STATISTIC].max())

    for player_id in range(5):

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
            fontsize=48,
            ax=ax,
            ha=CENTER,
            va=CENTER,
            str_format="{:.0%}",
            path_effects=path_eff,
        )

        plt.show()

        # Save the plot
        fig.savefig(
            f"{DATA_OUTPUTS_PATH}{PLAYER}_{player_id}_{RECEPTIONS_HEATMAP_NAME}",
            bbox_inches=TIGHT,
        )


def calcluate_player_level_values_for_receptions(receptions):

    # Calculate player-level values for receptions
    total_minutes_per_player = receptions.groupby(
        [PLAYER_ID, PLAYER_NAME, MATCH_ID], as_index=False, dropna=False
    )[MINUTES_PLAYED].max()
    total_minutes_per_player = total_minutes_per_player.groupby(
        [PLAYER_ID, PLAYER_NAME], as_index=False, dropna=False
    )[MINUTES_PLAYED].sum()

    total_receptions_per_player = receptions.groupby(
        [PLAYER_ID, PLAYER_NAME], as_index=False, dropna=False
    )[POSSESSION_INDEX].count()
    total_receptions_per_player = total_receptions_per_player.rename(
        columns={POSSESSION_INDEX: TOTAL_RECEPTIONS}
    )

    total_receptions_in_zone_14_and_17_per_player = (
        receptions[receptions[ZONE_14_AND_17] == True]
        .groupby([PLAYER_ID, PLAYER_NAME], as_index=False, dropna=False)[POSSESSION_INDEX]
        .count()
    )
    total_receptions_in_zone_14_and_17_per_player = (
        total_receptions_in_zone_14_and_17_per_player.rename(
            columns={POSSESSION_INDEX: TOTAL_RECEPTIONS_IN_ZONE_14_AND_17}
        )
    )

    total_receptions_in_final_third_per_player = (
        receptions[receptions[FINAL_THIRD] == True]
        .groupby([PLAYER_ID, PLAYER_NAME], as_index=False, dropna=False)[POSSESSION_INDEX]
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
        .groupby([PLAYER_ID, PLAYER_NAME], as_index=False, dropna=False)[POSSESSION_INDEX]
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
        .groupby([PLAYER_ID, PLAYER_NAME], as_index=False, dropna=False)[EXPECTED_GOALS]
        .agg(
            total_shots_in_zone_14_and_17=COUNT,
            total_expected_goals_in_zone_14_and_17=SUM,
            mean_expected_goals_per_shot_from_receptions_in_zone_14_and_17=MEAN,
            median_expected_goals_per_shot_from_receptions_in_zone_14_and_17=MEDIAN,
        )
    )

    average_event_score_per_reception_in_zone_14_and_17_per_player = (
        receptions[receptions[ZONE_14_AND_17] == True]
        .groupby([PLAYER_ID, PLAYER_NAME], as_index=False, dropna=False)[POSSESSION_EVENT_SCORE]
        .agg(
            mean_event_score_per_reception_in_zone_14_and_17=MEAN,
            median_event_score_per_reception_in_zone_14_and_17=MEDIAN,
        )
    )

    # Merge player-level values for receptions
    player_reception_metrics = total_minutes_per_player.merge(
        total_receptions_per_player, on=[PLAYER_ID, PLAYER_NAME], how=LEFT
    )
    player_reception_metrics = player_reception_metrics.merge(
        total_receptions_in_zone_14_and_17_per_player, on=[PLAYER_ID, PLAYER_NAME], how=LEFT
    )
    player_reception_metrics = player_reception_metrics.merge(
        total_receptions_in_final_third_per_player, on=[PLAYER_ID, PLAYER_NAME], how=LEFT
    )
    player_reception_metrics = player_reception_metrics.merge(
        total_shooting_opportunities_in_zone_14_and_17_per_player,
        on=[PLAYER_ID, PLAYER_NAME],
        how=LEFT,
    )
    player_reception_metrics = player_reception_metrics.merge(
        shots_in_zone_14_and_17_per_player, on=[PLAYER_ID, PLAYER_NAME], how=LEFT
    )
    player_reception_metrics = player_reception_metrics.merge(
        average_event_score_per_reception_in_zone_14_and_17_per_player,
        on=[PLAYER_ID, PLAYER_NAME],
        how=LEFT,
    )

    return player_reception_metrics


def output_metrics_for_receptions(receptions):

    player_reception_metrics = calcluate_player_level_values_for_receptions(receptions)

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
    player_reception_metrics[SHOOTING_OPPORTUNITIES_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17] = (
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
    player_reception_metrics[EXPECTED_GOALS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17] = (
        player_reception_metrics[TOTAL_EXPECTED_GOALS_IN_ZONE_14_AND_17]
        / player_reception_metrics[MINUTES_PLAYED]
        * 90
    )

    player_reception_metrics = player_reception_metrics[
        [PLAYER_ID,
        PLAYER_NAME,
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
        MEDIAN_EVENT_SCORE_PER_RECEPTION_IN_ZONE_14_AND_17]
    ]

    # Save the reception metrics
    player_reception_metrics.to_csv(
        f"{DATA_OUTPUTS_PATH}{RECEPTIONS_METRICS_NAME}",
        index=False,
    )

    return player_reception_metrics


def receptions_analysis():

    # Get the receptions dataset
    posessions = possesions_dataset()

    receptions = posessions[posessions[TYPE_NAME] == BALL_RECEIPT]

    # Loop through each plaer and create heatmap of receptions
    plot_receptions_on_pitch(receptions)

    # Create output metrics for receptions
    receptions_metrics = output_metrics_for_receptions(receptions)

    return receptions_metrics

if __name__ == "__main__":
    receptions_analysis()
