import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from adjustText import adjust_text
from mplsoccer import Pitch, VerticalPitch

from src.config import *


def possesions_dataset():

    # Read in processed events data
    events_df = pd.read_csv(f"{DATA_PROCESSED_PATH}{TRANSFOMED_EVENTS_DATA_NAME}")

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


def calcluate_player_level_values_for_receptions(
    receptions: pd.DataFrame, subset: list
):

    # Calculate player-level values for receptions
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

    # Merge player-level values for receptions
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


def output_metrics_for_receptions(
    receptions: pd.DataFrame, subset=[PLAYER_ID, PLAYER_NAME]
):

    player_reception_metrics = calcluate_player_level_values_for_receptions(
        receptions, subset
    )

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


def calcluate_player_level_values_for_goals(posessions: pd.DataFrame, subset):

    # Calculate player-level values for goals
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


def output_metrics_for_goals(posessions: pd.DataFrame, subset=[PLAYER_ID, PLAYER_NAME]):

    player_goal_metrics = calcluate_player_level_values_for_goals(posessions, subset)

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


def calculate_team_metrics(matches_df, team_id):

    # Filter matches for team
    team_matches = matches_df[
        (matches_df[HOME_TEAM_ID] == team_id) | (matches_df[AWAY_TEAM_ID] == team_id)
    ]

    # Calculate team and opponent ELO ratings
    team_matches.loc[team_matches[HOME_TEAM_ID] == team_id, TEAM_ELO] = team_matches[
        HOME_ELO
    ]
    team_matches.loc[team_matches[AWAY_TEAM_ID] == team_id, TEAM_ELO] = team_matches[
        AWAY_ELO
    ]
    team_matches.loc[team_matches[HOME_TEAM_ID] == team_id, OPPOSITION_ELO] = (
        team_matches[AWAY_ELO]
    )
    team_matches.loc[team_matches[AWAY_TEAM_ID] == team_id, OPPOSITION_ELO] = (
        team_matches[HOME_ELO]
    )

    # Calculate team and opponent expected goals
    team_matches.loc[team_matches[HOME_TEAM_ID] == team_id, TEAM_EXPECTED_GOALS] = (
        team_matches[HOME_TEAM_XG]
    )
    team_matches.loc[team_matches[AWAY_TEAM_ID] == team_id, TEAM_EXPECTED_GOALS] = (
        team_matches[AWAY_TEAM_XG]
    )
    team_matches.loc[team_matches[HOME_TEAM_ID] == team_id, OPPONENT_EXPECTED_GOALS] = (
        team_matches[AWAY_TEAM_XG]
    )
    team_matches.loc[team_matches[AWAY_TEAM_ID] == team_id, OPPONENT_EXPECTED_GOALS] = (
        team_matches[HOME_TEAM_XG]
    )

    # Calculate team-level metrics
    team_matches[DIFFERENCE_TO_OPPONENT_ELO] = (
        team_matches[TEAM_ELO] - team_matches[OPPOSITION_ELO]
    )
    team_matches[DIFFERENCE_TO_OPPONENT_EXPECTED_GOALS] = (
        team_matches[TEAM_EXPECTED_GOALS] - team_matches[OPPONENT_EXPECTED_GOALS]
    )

    # Calcluate mean and median values for team-level metrics and add to data table with team ID
    team_matches[TEAM_ID] = team_id

    mean_team_elo_rating = team_matches.groupby(TEAM_ID, as_index=False, dropna=False)[
        TEAM_ELO
    ].agg(
        mean_team_elo_rating=MEAN,
        median_team_elo_rating=MEDIAN,
    )

    mean_opposition_elo_rating = team_matches.groupby(
        TEAM_ID, as_index=False, dropna=False
    )[OPPOSITION_ELO].agg(
        mean_opposition_elo_rating=MEAN,
        median_opposition_elo_rating=MEDIAN,
    )

    mean_expected_goals = team_matches.groupby(TEAM_ID, as_index=False, dropna=False)[
        TEAM_EXPECTED_GOALS
    ].mean()
    mean_expected_goals = mean_expected_goals.rename(
        columns={TEAM_EXPECTED_GOALS: MEAN_EXPECTED_GOALS}
    )

    difference_to_opponent_elo = team_matches.groupby(
        TEAM_ID, as_index=False, dropna=False
    )[DIFFERENCE_TO_OPPONENT_ELO].agg(
        mean_difference_to_opponent_elo=MEAN,
        median_difference_to_opponent_elo=MEDIAN,
    )

    difference_to_opponent_expected_goals = team_matches.groupby(
        TEAM_ID, as_index=False, dropna=False
    )[DIFFERENCE_TO_OPPONENT_EXPECTED_GOALS].agg(
        mean_difference_to_opponent_expected_goals=MEAN,
        median_difference_to_opponent_expected_goals=MEDIAN,
    )

    team_metrics = mean_team_elo_rating.merge(
        mean_opposition_elo_rating, on=TEAM_ID, how=LEFT
    )
    team_metrics = team_metrics.merge(mean_expected_goals, on=TEAM_ID, how=LEFT)
    team_metrics = team_metrics.merge(difference_to_opponent_elo, on=TEAM_ID, how=LEFT)
    team_metrics = team_metrics.merge(
        difference_to_opponent_expected_goals, on=TEAM_ID, how=LEFT
    )

    return team_metrics


def output_team_metrics():

    # Read in match data
    matches_df = pd.read_csv(f"{DATA_RAW_PATH}{MATCHES_DATA_NAME}")

    # Calculate team metrics for each team
    team_metrics = pd.DataFrame()
    for team_id in TEAM_IDS:
        team_metrics = pd.concat(
            [team_metrics, calculate_team_metrics(matches_df, team_id)],
            ignore_index=True,
        )

    # Save the team metrics
    team_metrics.to_csv(f"{DATA_OUTPUTS_PATH}{TEAM_METRICS_NAME}", index=False)

    return team_metrics


def create_receptions_slide_table(player_receptions_metrics):

    # Select required metrics
    receptions_slide_table = player_receptions_metrics[
        [
            PLAYER_NAME,
            RECEPTIONS_PER_90,
            RECEPTIONS_IN_ZONE_14_AND_17_PER_90,
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


def create_scatter_plot_of_two_metrics(df: pd.DataFrame, x_metric: str, y_metric: str):

    df[PLAYER_TEAM_COMBO] = df[PLAYER_NAME] + " (" + df[TEAM_NAME] + ")"

    # Create a color palette based on the unique combinations
    unique_combinations = df[PLAYER_TEAM_COMBO].unique()
    palette = sns.color_palette("husl", len(unique_combinations))
    color_mapping = {combo: palette[i] for i, combo in enumerate(unique_combinations)}

    # Map colors to each point in the DataFrame
    df[COLOR] = df[PLAYER_TEAM_COMBO].map(color_mapping)

    # Create a figure
    fig, ax = plt.subplots(figsize=(15, 5.5))

    # Determine axis limits for boundary checks
    x_min, x_max = df[x_metric].min(), df[x_metric].max()
    y_min, y_max = df[y_metric].min(), df[y_metric].max()
    x_margin = (x_max - x_min) * 0.02  # Small margin for x
    y_margin = (y_max - y_min) * 0.02  # Small margin for y

    # Plot the scatter plot with colors
    for _, row in df.iterrows():
        ax.scatter(
            row[x_metric],
            row[y_metric],
            color=row[COLOR],
            edgecolor=WHITE,
            s=100,
            linewidth=2,
            zorder=2,
        )

        # Calculate dynamic offsets to avoid overlapping and keep in bounds
        x_offset = x_margin if row[x_metric] < (x_max - x_margin) else -x_margin
        y_offset = y_margin if row[y_metric] < (y_max - y_margin) else -y_margin

        # Add text with offset
        ax.text(
            row[x_metric] + x_offset,
            row[y_metric] + y_offset,
            row[PLAYER_TEAM_COMBO],
            fontsize=16,
            ha=LEFT if x_offset > 0 else RIGHT,  # Adjust alignment dynamically
            va=BOTTOM if y_offset > 0 else TOP,
            zorder=3,
        )
    
    ax.tick_params(axis='both', labelsize=16)

    # Save the scatter plot
    fig.tight_layout()
    fig.savefig(
        f"{DATA_OUTPUTS_PATH}{x_metric}_vs_{y_metric}_{SCATTER_PLOT_NAME}",
        bbox_inches=TIGHT,
    )


def create_scatter_plot_for_goal_scoring_metrics(player_team_goals_metrics):

    create_scatter_plot_of_two_metrics(
        player_team_goals_metrics,
        SHOOTING_OPPORTUNITIES_PER_POSSESSION,
        SHOTS_PER_POSSESSION,
    )
    create_scatter_plot_of_two_metrics(
        player_team_goals_metrics, SHOTS_PER_90, MEAN_EXPECTED_GOALS_PER_SHOT
    )
    create_scatter_plot_of_two_metrics(
        player_team_goals_metrics, EXPECTED_GOALS_PER_90, GOALS_PER_90
    )


def analysis():

    # Get the receptions dataset
    posessions = possesions_dataset()

    receptions = posessions[posessions[TYPE_NAME] == BALL_RECEIPT]

    # Loop through each plaer and create heatmap of receptions
    plot_receptions_on_pitch(receptions)

    # Create output metrics for receptions
    player_receptions_metrics = output_metrics_for_receptions(
        receptions, [PLAYER_ID, PLAYER_NAME]
    )
    player_team_receptions_metrics = output_metrics_for_receptions(
        receptions, [PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_NAME]
    )

    # Create output metrics for goals
    player_goals_metrics = output_metrics_for_goals(
        posessions, [PLAYER_ID, PLAYER_NAME]
    )
    player_team_goals_metrics = output_metrics_for_goals(
        posessions, [PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_NAME]
    )

    # Create team-level metrics
    team_metrics = output_team_metrics()

    # Create table for receptions slide
    create_receptions_slide_table(player_receptions_metrics)

    # Create scatter plots for goal scoring metrics
    create_scatter_plot_for_goal_scoring_metrics(player_team_goals_metrics)


if __name__ == "__main__":
    analysis()
