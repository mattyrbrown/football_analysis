import pandas as pd
import seaborn as sns

from src.config import *
from src.chart_utils import *


def player_team_dataset() -> pd.DataFrame:
    """
    From the merged events and match data, create a dataset that contains the team and opposition ELO rating and the ELO
    rating difference for each match that a target has played in. Each row in the dataset represents a unique match that a
    target has played in

    Returns:
        events_matches_df (pd.DataFrame): A dataset containing the team and opposition ELO rating and the ELO rating
        difference for each match that a target has played in
    """

    # Read in merged and transformed events and match data
    events_matches_df = pd.read_csv(
        f"{DATA_PROCESSED_PATH}{MERGED_EVENTS_MATCH_LEVEL_DATA_NAME}"
    )

    # Group data by player and match and take first team and opposition ELO rating and the ELO rating difference for each match
    # that a target has played in. The first can be taken as these values are the same for each event in a match and player
    # combination
    events_matches_df = events_matches_df.groupby(
        [PLAYER_ID, PLAYER_NAME, MATCH_ID], as_index=False, dropna=False
    )[[TEAM_ELO, OPPOSITION_ELO, ELO_DIFFERENCE]].first()

    # Return the dataset
    return events_matches_df


def possesions_dataset() -> pd.DataFrame:
    """
    From the merged events and match data, create a dataset that has a single row for each possession by targets. In the
    merged events and match data, there are single event possessions labeled for pressures, duels that are not won and
    penalties. These are disregard as possessions for creation of this possessions dataset. The dataset displays for each
    possession information on the match, team, player, start event type, start location, end event type, end location. In
    addition, for each possession, the dataset determines if it started in zone 14 and 17 or the final third of the pitch,
    if it ended in a shot or an assist of a shot, and the total sum of event value for all events in the possession.

    Returns:
        possessions (pd.DataFrame): A dataset containing a single row for each possession by targets
    """

    # Read in merged and transformed events and match data
    events_df = pd.read_csv(
        f"{DATA_PROCESSED_PATH}{MERGED_EVENTS_MATCH_LEVEL_DATA_NAME}"
    )

    # Filter out possessions labeled for pressures, duels that are not won and penalties as these are not genuine possessions
    possessions = events_df[
        (events_df[TYPE_NAME] != PRESSURE)
        & (
            (events_df[DUEL_OUTCOME_NAME] == WON)
            | events_df[DUEL_OUTCOME_NAME].isnull()
        )
        & (events_df[START_OF_POSSESSION] == True)
        & (events_df[PENALTY] == False)
    ]

    # Keep relevant columns for possessions
    possessions = possessions[
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

    # Label possessions that start in zone 14 and 17
    possessions[ZONE_14_AND_17] = False
    possessions.loc[
        (possessions[LOCATION_X] >= X_FINAL_THIRD_MIN)
        & (possessions[LOCATION_Y] >= Y_ZONE_LINE_MIN)
        & (possessions[LOCATION_Y] <= Y_ZONE_LINE_MAX),
        ZONE_14_AND_17,
    ] = True

    # Label possessions that start in the final third
    possessions[FINAL_THIRD] = False
    possessions.loc[
        possessions[LOCATION_X] >= X_FINAL_THIRD_MIN,
        FINAL_THIRD,
    ] = True

    # Find unique possession indexes remaining after filtering and then filter events data for these possession indexes
    possession_indexes = possessions[POSSESSION_INDEX].unique()
    possessions_events = events_df[events_df[POSSESSION_INDEX].isin(possession_indexes)]

    # Using the events data, create dataset that for each possession shows the end event type, end location, if the possession
    # ended in a goal or pass assisting shot, and the expected goals for shots at the end of a possession
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

    # For the possession outcome dataset, create field to show if the possession ended in a shot or pass assisting a shot
    outcome_of_posessions[SHOOTING_OPPORTUNITY] = False
    outcome_of_posessions.loc[
        (outcome_of_posessions[OUTCOME_TYPE_NAME] == SHOT)
        | (outcome_of_posessions[PASS_ASSISTED_SHOT_ID].notnull()),
        SHOOTING_OPPORTUNITY,
    ] = True

    # Using the events data, create dataset that for each possession shows the total sum of event value for all events in the
    # possession
    event_score_possessions = possessions_events.groupby(
        POSSESSION_INDEX, as_index=False, dropna=False
    )[EVENT_VALUE].agg(possession_event_score=SUM, events=COUNT)

    # Merge the possessions dataset with the outcome of possessions and event score of possessions datasets to create a single
    # dataset for possessions
    possessions = possessions.merge(
        outcome_of_posessions,
        on=POSSESSION_INDEX,
        how=LEFT,
    )
    possessions = possessions.merge(
        event_score_possessions,
        on=POSSESSION_INDEX,
        how=LEFT,
    )

    # Return the possessions dataset
    return possessions


def calcluate_values_for_receptions(
    receptions: pd.DataFrame, subset: list
) -> pd.DataFrame:
    """
    From the receptions dataset, calculate total values needed to determine reception metrics by the proposed subset provided.
    A dataframe is returned with the total values calculated segmented by the provided subset. The total values calcluated
    are:
    - Total minutes played
    - Total receptions
    - Total receptions in zone 14 and 17
    - Total receptions in the final third
    - Total shots and shot assists from receptions in zone 14 and 17
    - Total shots from receptions in zone 14 and 17
    - Total expected goals from receptions in zone 14 and 17
    - Mean expected goals per shot from receptions in zone 14 and 17
    - Median expected goals per shot from receptions in zone 14 and 17
    - Mean event score per reception in zone 14 and 17
    - Median event score per reception in zone 14 and 17

    Args:
        receptions (pd.DataFrame): A dataset containing receptions data
        subset (list): A list of columns to segment the dataset by

    Returns:
        player_reception_metrics (pd.DataFrame): A dataset containing the total values for the reception metrics segmented by
        the provided subset
    """

    # Calcluate total minutes played by subset
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

    # Calculate total receptions by subset
    total_receptions_per_player = receptions.groupby(
        subset, as_index=False, dropna=False
    )[POSSESSION_INDEX].count()
    total_receptions_per_player = total_receptions_per_player.rename(
        columns={POSSESSION_INDEX: TOTAL_RECEPTIONS}
    )

    # Calculate total receptions in zone 14 and 17 by subset
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

    # Calculate total receptions in the final third by subset
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

    # Calculate total shots and shot assists from receptions in zone 14 and 17 by subset
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

    # Calculate total shots from receptions in zone 14 and 17, total expected goals from receptions in zone 14 and 17, mean
    # expected goals per shot from receptions in zone 14 and 17, and median expected goals per shot from receptions in zone 14
    # and 17 by subset
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

    # Calculate mean event score per reception in zone 14 and 17 and median event score per reception in zone 14 and 17 by
    # subset
    average_event_score_per_reception_in_zone_14_and_17_per_player = (
        receptions[receptions[ZONE_14_AND_17] == True]
        .groupby(subset, as_index=False, dropna=False)[POSSESSION_EVENT_SCORE]
        .agg(
            mean_event_score_per_reception_in_zone_14_and_17=MEAN,
            median_event_score_per_reception_in_zone_14_and_17=MEDIAN,
        )
    )

    # Merge total values for receptions into a single dataset
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

    # Return the dataset containing the total values for the reception metrics
    return player_reception_metrics


def calculate_reception_metrics(
    player_reception_metrics: pd.DataFrame, subset: list[str]
) -> pd.DataFrame:
    """
    From the dataset containing the total values for the reception metrics, calculate the output metrics for receptions by the
    provided segmentation of the dataset. A dataframe is returned with the output metrics segmented by the provided
    subset. The output metrics calculated are:
    - Receptions per 90 minutes
    - Receptions in zone 14 and 17 per 90 minutes
    - Percentage of receptions in zone 14 and 17
    - Receptions in the final third per 90 minutes
    - Percentage of receptions in the final third
    - Shots and shot assists per reception in zone 14 and 17
    - Shots and shot assists per 90 minutes from receptions in zone 14 and 17
    - Shots per reception in zone 14 and 17
    - Shots per 90 minutes from receptions in zone 14 and 17
    - Expected goals per 90 minutes from receptions in zone 14 and 17
    - Mean expected goals per shot from receptions in zone 14 and 17
    - Median expected goals per shot from receptions in zone 14 and 17
    - Mean event score per reception in zone 14 and 17
    - Median event score per reception in zone 14 and 17
    If the subset contains MatchID, the output includes the total minutes played, opposition ELO weighting, and ELO difference
    weighting to allow for normalisation of the metrics.

    Args:
        player_reception_metrics (pd.DataFrame): A dataset containing the total values for the reception metrics
        subset (list): A list of columns to segment the dataset by

    Returns:
        player_reception_metrics (pd.DataFrame): A dataset containing the output metrics for receptions by the provided
        segmentation
    """

    # Calculate receptions per 90 minutes
    player_reception_metrics[RECEPTIONS_PER_90] = (
        player_reception_metrics[TOTAL_RECEPTIONS]
        / player_reception_metrics[MINUTES_PLAYED]
        * 90
    )

    # Calculate receptions in zone 14 and 17 per 90 minutes
    player_reception_metrics[RECEPTIONS_IN_ZONE_14_AND_17_PER_90] = (
        player_reception_metrics[TOTAL_RECEPTIONS_IN_ZONE_14_AND_17]
        / player_reception_metrics[MINUTES_PLAYED]
        * 90
    )

    # Calculate percentage of receptions in zone 14 and 17
    player_reception_metrics[PERCENTAGE_RECEPTIONS_IN_ZONE_14_AND_17] = (
        player_reception_metrics[TOTAL_RECEPTIONS_IN_ZONE_14_AND_17]
        / player_reception_metrics[TOTAL_RECEPTIONS]
        * 100
    )

    # Calculate receptions in the final third per 90 minutes
    player_reception_metrics[RECEPTIONS_IN_FINAL_THIRD_PER_90] = (
        player_reception_metrics[TOTAL_RECEPTIONS_IN_FINAL_THIRD]
        / player_reception_metrics[MINUTES_PLAYED]
        * 90
    )

    # Calculate percentage of receptions in the final third
    player_reception_metrics[PERCENTAGE_RECEPTIONS_IN_FINAL_THIRD] = (
        player_reception_metrics[TOTAL_RECEPTIONS_IN_FINAL_THIRD]
        / player_reception_metrics[TOTAL_RECEPTIONS]
        * 100
    )

    # Calculate shots and shot assists per reception in zone 14 and 17
    player_reception_metrics[SHOOTING_OPPORTUNITIES_PER_RECEPTION_IN_ZONE_14_AND_17] = (
        player_reception_metrics[TOTAL_SHOOTING_OPPORTUNITIES_IN_ZONE_14_AND_17]
        / player_reception_metrics[TOTAL_RECEPTIONS_IN_ZONE_14_AND_17]
    )

    # Calculate shots and shot assists per 90 minutes from receptions in zone 14 and 17
    player_reception_metrics[
        SHOOTING_OPPORTUNITIES_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17
    ] = (
        player_reception_metrics[TOTAL_SHOOTING_OPPORTUNITIES_IN_ZONE_14_AND_17]
        / player_reception_metrics[MINUTES_PLAYED]
        * 90
    )

    # Calculate shots per reception in zone 14 and 17
    player_reception_metrics[SHOTS_PER_RECEPTION_IN_ZONE_14_AND_17] = (
        player_reception_metrics[TOTAL_SHOTS_IN_ZONE_14_AND_17]
        / player_reception_metrics[TOTAL_RECEPTIONS_IN_ZONE_14_AND_17]
    )

    # Calculate shots per 90 minutes from receptions in zone 14 and 17
    player_reception_metrics[SHOTS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17] = (
        player_reception_metrics[TOTAL_SHOTS_IN_ZONE_14_AND_17]
        / player_reception_metrics[MINUTES_PLAYED]
        * 90
    )

    # Calculate expected goals per 90 minutes from receptions in zone 14 and 17
    player_reception_metrics[
        EXPECTED_GOALS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17
    ] = (
        player_reception_metrics[TOTAL_EXPECTED_GOALS_IN_ZONE_14_AND_17]
        / player_reception_metrics[MINUTES_PLAYED]
        * 90
    )

    # Return dataset containing the output metrics for receptions by the provided segmentation
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

    # Return the dataset containing the output metrics for receptions by the provided segmentation
    return player_reception_metrics


def output_metrics_for_receptions(
    receptions: pd.DataFrame, subset: list[str] = [PLAYER_ID, PLAYER_NAME]
) -> pd.DataFrame:
    """
    From the receptions dataset, calculate receptions output metrics by a provided segmentation of the dataset. The receptions
    output metrics data table is saved to a csv file in the data/outputs folder.

    Args:
        receptions (pd.DataFrame): A dataset containing receptions data
        subset (list): A list of columns to segment the dataset by

    Returns:
        player_reception_metrics (pd.DataFrame): A dataset containing the output metrics for receptions by the provided
        segmentation
    """

    # Calculate total values for forming reception metrics
    player_reception_metrics = calcluate_values_for_receptions(receptions, subset)

    # Calculate reception metrics from total values
    player_reception_metrics = calculate_reception_metrics(
        player_reception_metrics, subset
    )

    # Save the reception output metrics as a csv file in the data/outputs folder
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

    # Return the reception output metrics
    return player_reception_metrics


def normalise_metrics(df: pd.DataFrame, metric_columns: list) -> pd.DataFrame:
    """
    Normalise the provided metric columns in a dataset by the opposition ELO weighting, ELO difference weighting. Aggregate
    the normalised metrics to player level using a weighted average based on minutes played. The aggregated normalised metrics
    are then each scaled using 0-100 scale with a value of 100 representing the maximum value of the metric across all players.
    The aggregated normalised metric scores are returned in a dataset.

    Args:
        df (pd.DataFrame): A dataset containing the metrics to normalise
        metric_columns (list): A list of metric columns to normalise

    Returns:
        df (pd.DataFrame): A dataset containing the aggregated normalised metrics scores for each player
    """

    # Filter out rows where opposition ELO weighting and ELO difference weighting are null and so the metrics cannot be
    # normalised
    df = df[
        (df[OPPOSITION_ELO_WEIGHTING].notnull())
        & (df[ELO_DIFFERENCE_WEIGHTING].notnull())
    ]

    # Normalise the metric columns by the opposition ELO weighting and ELO difference weighting, and then multiply by
    # minutes played to support weighted average aggregation
    for column in metric_columns:
        df[column] = (
            df[column] * df[OPPOSITION_ELO_WEIGHTING] * df[ELO_DIFFERENCE_WEIGHTING]
        )
        df[column] = df[column] * df[MINUTES_PLAYED]

    # Aggregate the normalised metrics multiplied by minutes played to player level along with the total minutes played
    df = df.groupby([PLAYER_ID, PLAYER_NAME], as_index=False, dropna=False)[
        metric_columns + [MINUTES_PLAYED]
    ].sum()

    # For each metric, divide the aggregated normalised metric multiplied by minutes played by the total minutes played to
    # find the weighted average of the normalised metric at player level. Then scale the aggregated normalised metric to a
    # 0-100 scale with 100 representing the maximum value of the metric across all players
    for column in metric_columns:
        df[column] = df[column] / df[MINUTES_PLAYED]
        df[column] = (df[column] * 100) / df[column].max()

    # Return the dataset containing the aggregated normalised metrics scores for each player
    return df


def output_normalised_metrics_for_receptions(receptions: pd.DataFrame) -> pd.DataFrame:
    """
    From the receptions dataset, calculate normalised output reception metrics for each player. Metrics are normalised at a
    match and player level by the opposition ELO weighting and ELO difference weighting. The normalised metrics are then
    aggregated to player level using a weighted average based on minutes played. Finally, the aggregated normalised metrics
    are each scaled to a 0-100 scale with 100 representing the maximum value of the metric across all players. A dataset is
    returned with the scaled normalised output reception metrics for each player. The dataset is also saved to a csv file in
    the data/outputs folder.

    Args:
        receptions (pd.DataFrame): A dataset containing receptions data

    Returns:
        player_normalised_reception_metrics (pd.DataFrame): A dataset containing the scaled normalised output reception
        metrics for each player
    """

    # Calculate total values by player and match for forming reception metrics
    player_normalised_reception_metrics = calcluate_values_for_receptions(
        receptions, [MATCH_ID, PLAYER_ID, PLAYER_NAME]
    )

    # Calculate reception metrics by player and match from total values
    player_normalised_reception_metrics = calculate_reception_metrics(
        player_normalised_reception_metrics, [MATCH_ID, PLAYER_ID, PLAYER_NAME]
    )

    # Determine metric columns to normalise
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

    # Normalise the metrics by opposition ELO weighting and ELO difference weighting and aggregate to player level using a
    # weighted average based on minutes played. Scale the aggregated normalised metrics to a 0-100 scale with 100 representing
    # the maximum value of the metric across all players
    player_normalised_reception_metrics = normalise_metrics(
        player_normalised_reception_metrics, metric_columns
    )

    # Save the normalised reception metrics as a csv file in the data/outputs folder
    player_normalised_reception_metrics.to_csv(
        f"{DATA_OUTPUTS_PATH}{PLAYER}_{RECEPTIONS_NORMALISED_METRICS_NAME}",
        index=False,
    )

    # Return the scaled normalised output reception metrics for each player
    return player_normalised_reception_metrics


def calcluate_values_for_goals(
    posessions: pd.DataFrame, subset: list[str]
) -> pd.DataFrame:
    """
    From the posessions dataset, calculate total values needed to determine goal metrics by the proposed subset provided. A
    dataframe is returned with the total values calculated segmented by the provided subset. The total values calcluated are:
    - Total minutes played
    - Total posessions
    - Total shots
    - Total shots and shot assists
    - Total goals
    - Total expected goals
    - Mean expected goals per shot
    - Median expected goals per shot

    Args:
        posessions (pd.DataFrame): A dataset containing posessions data
        subset (list): A list of columns to segment the dataset by

    Returns:
        player_goal_metrics (pd.DataFrame): A dataset containing the total values for the goal metrics segmented by the provided
        subset
    """

    # Calculate total minutes played by subset
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

    # Calculate total posessions by subset
    posessions_per_player = posessions.groupby(subset, as_index=False, dropna=False)[
        POSSESSION_INDEX
    ].count()
    posessions_per_player = posessions_per_player.rename(
        columns={POSSESSION_INDEX: TOTAL_POSSESSIONS}
    )

    # Calculate total shots by subset
    total_shots_per_player = (
        posessions[posessions[OUTCOME_TYPE_NAME] == SHOT]
        .groupby(subset, as_index=False, dropna=False)[POSSESSION_INDEX]
        .count()
    )
    total_shots_per_player = total_shots_per_player.rename(
        columns={POSSESSION_INDEX: TOTAL_SHOTS}
    )

    # Calculate total shots and shot assists by subset
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

    # Calculate total goals by subset
    total_goals_per_player = (
        posessions[posessions[GOAL] == True]
        .groupby(subset, as_index=False, dropna=False)[POSSESSION_INDEX]
        .count()
    )
    total_goals_per_player = total_goals_per_player.rename(
        columns={POSSESSION_INDEX: TOTAL_GOALS}
    )

    # Calculate total expected goals by subset
    expected_goals_per_player = (
        posessions[posessions[OUTCOME_TYPE_NAME] == SHOT]
        .groupby(subset, as_index=False, dropna=False)[EXPECTED_GOALS]
        .agg(
            total_expected_goals=SUM,
            mean_expected_goals_per_shot=MEAN,
            median_expected_goals_per_shot=MEDIAN,
        )
    )

    # Merge total values for goals into a single dataset
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

    # Return the dataset containing the total values for the goal metrics
    return player_goal_metrics


def calculate_goals_metrics(
    player_goal_metrics: pd.DataFrame, subset: list[str]
) -> pd.DataFrame:
    """
    From the dataset containing the total values for the goal metrics, calculate the output metrics for goals by the provided
    segmentation of the dataset. A dataframe is returned with the output metrics segmented by the provided subset. The output
    metrics calculated are:
    - Shots and shot assists per 90 minutes
    - Shots and shot assists per posession
    - Shots per 90 minutes
    - Shots per posession
    - Goals per 90 minutes
    - Goals per shot
    - Goals per posession
    - Expected goals per 90 minutes
    - Mean expected goals per shot
    - Median expected goals per shot
    - Ratio of goals to expected goals
    If the subset contains MatchID, the output includes the total minutes played, opposition ELO weighting, and ELO difference
    weighting to allow for normalisation of the metrics.

    Args:
        player_goal_metrics (pd.DataFrame): A dataset containing the total values for the goal metrics
        subset (list): A list of columns to segment the dataset by

    Returns:
        player_goal_metrics (pd.DataFrame): A dataset containing the output metrics for goals by the provided segmentation
    """

    # Calculate shots and shot assists per 90 minutes
    player_goal_metrics[SHOOTING_OPPORTUNITIES_PER_90] = (
        player_goal_metrics[TOTAL_SHOOTING_OPPORTUNITIES]
        / player_goal_metrics[MINUTES_PLAYED]
        * 90
    )

    # Calculate shots and shot assists per posession
    player_goal_metrics[SHOOTING_OPPORTUNITIES_PER_POSSESSION] = (
        player_goal_metrics[TOTAL_SHOOTING_OPPORTUNITIES]
        / player_goal_metrics[TOTAL_POSSESSIONS]
    )

    # Calculate shots per 90 minutes
    player_goal_metrics[SHOTS_PER_90] = (
        player_goal_metrics[TOTAL_SHOTS] / player_goal_metrics[MINUTES_PLAYED] * 90
    )

    # Calculate shots per posession
    player_goal_metrics[SHOTS_PER_POSSESSION] = (
        player_goal_metrics[TOTAL_SHOTS] / player_goal_metrics[TOTAL_POSSESSIONS]
    )

    # Calculate goals per 90 minutes
    player_goal_metrics[GOALS_PER_90] = (
        player_goal_metrics[TOTAL_GOALS] / player_goal_metrics[MINUTES_PLAYED] * 90
    )

    # Calculate goals per shot
    player_goal_metrics[GOALS_PER_SHOT] = (
        player_goal_metrics[TOTAL_GOALS] / player_goal_metrics[TOTAL_SHOTS]
    )

    # Calculate goals per posession
    player_goal_metrics[GOALS_PER_POSSESSION] = (
        player_goal_metrics[TOTAL_GOALS] / player_goal_metrics[TOTAL_POSSESSIONS]
    )

    # Calculate expected goals per 90 minutes
    player_goal_metrics[EXPECTED_GOALS_PER_90] = (
        player_goal_metrics[TOTAL_EXPECTED_GOALS]
        / player_goal_metrics[MINUTES_PLAYED]
        * 90
    )

    # Calculate ratio of goals to expected goals
    player_goal_metrics[RATIO_OF_GOALS_TO_EXPECTED_GOALS] = (
        player_goal_metrics[TOTAL_GOALS] / player_goal_metrics[TOTAL_EXPECTED_GOALS]
    )

    # Return dataset containing the output metrics for goals by the provided segmentation
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

    # Return the dataset containing the output metrics for goals by the provided segmentation
    return player_goal_metrics


def output_metrics_for_goals(
    posessions: pd.DataFrame, subset: list[str] = [PLAYER_ID, PLAYER_NAME]
) -> pd.DataFrame:
    """
    From the posessions dataset, calculate goal output metrics by a provided segmentation of the dataset. The goal output
    metrics data table is saved to a csv file in the data/outputs folder.

    Args:
        posessions (pd.DataFrame): A dataset containing posessions data
        subset (list): A list of columns to segment the dataset by

    Returns:
        player_goal_metrics (pd.DataFrame): A dataset containing the output metrics for goals by the provided segmentation
    """

    # Calculate total values for forming goal metrics
    player_goal_metrics = calcluate_values_for_goals(posessions, subset)

    # Calculate goal metrics from total values
    player_goal_metrics = calculate_goals_metrics(player_goal_metrics, subset)

    # Save the goal output metrics as a csv file in the data/outputs folder
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

    # Return the goal output metrics
    return player_goal_metrics


def output_normalised_metrics_for_goals(possessions: pd.DataFrame) -> pd.DataFrame:
    """
    From the possessions dataset, calculate normalised output goal metrics for each player. Metrics are normalised at a match
    and player level by the opposition ELO weighting and ELO difference weighting. The normalised metrics are then aggregated
    to player level using a weighted average based on minutes played. Finally, the aggregated normalised metrics are each
    scaled to a 0-100 scale with 100 representing the maximum value of the metric across all players. A dataset is returned
    with the scaled normalised output goal metrics for each player. The dataset is also saved to a csv file in the
    data/outputs folder.

    Args:
        possessions (pd.DataFrame): A dataset containing possessions data

    Returns:
        player_normalised_goals_metrics (pd.DataFrame): A dataset containing the scaled normalised output goal metrics for
        each player
    """

    # Calculate total values by player and match for forming goal metrics
    player_normalised_goals_metrics = calcluate_values_for_goals(
        possessions, [MATCH_ID, PLAYER_ID, PLAYER_NAME]
    )

    # Calculate goal metrics by player and match from total values
    player_normalised_goals_metrics = calculate_goals_metrics(
        player_normalised_goals_metrics, [MATCH_ID, PLAYER_ID, PLAYER_NAME]
    )

    # Determine metric columns to normalise
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

    # Normalise the metrics by opposition ELO weighting and ELO difference weighting and aggregate to player level using a
    # weighted average based on minutes played. Scale the aggregated normalised metrics to a 0-100 scale with 100 representing
    # the maximum value of the metric across all players
    player_normalised_goals_metrics = normalise_metrics(
        player_normalised_goals_metrics, metric_columns
    )

    # Save the normalised goal metrics as a csv file in the data/outputs folder
    player_normalised_goals_metrics.to_csv(
        f"{DATA_OUTPUTS_PATH}{PLAYER}_{GOALS_NORMALISED_METRICS_NAME}",
        index=False,
    )

    # Return the scaled normalised output goal metrics for each player
    return player_normalised_goals_metrics


def create_receptions_slide_table(
    player_receptions_metrics, normalised_player_receptions_metrics
):
    """
    Create a table that can be used to present the key metrics for receptions within the final presentation. Using the tables
    created for player receptions metrics and normalised player receptions metrics, the table will contain the following:
    - Player name
    - Receptions in zone 14 and 17 per 90 minutes
    - Normalised receptions in zone 14 and 17 per 90 minutes
    - Percentage of receptions in zone 14 and 17
    - Percentage of receptions in the final third
    The final table is saved as a csv file in the data/outputs folder.

    Args:
        player_receptions_metrics (pd.DataFrame): A dataset containing reception metrics for each player
        normalised_player_receptions_metrics (pd.DataFrame): A dataset containing normalised reception metrics for each player
    """

    # Obtain the key metrics from receptions data at player level
    receptions_slide_table = player_receptions_metrics[
        [
            PLAYER_NAME,
            RECEPTIONS_IN_ZONE_14_AND_17_PER_90,
            PERCENTAGE_RECEPTIONS_IN_ZONE_14_AND_17,
            PERCENTAGE_RECEPTIONS_IN_FINAL_THIRD,
        ]
    ]

    # Merge the key normalised receptions metrics at player level
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

    # Reorder the columns
    receptions_slide_table = receptions_slide_table[
        [
            PLAYER_NAME,
            RECEPTIONS_IN_ZONE_14_AND_17_PER_90,
            f"{NORMALISED}_{RECEPTIONS_IN_ZONE_14_AND_17_PER_90}",
            PERCENTAGE_RECEPTIONS_IN_ZONE_14_AND_17,
            PERCENTAGE_RECEPTIONS_IN_FINAL_THIRD,
        ]
    ]

    # Transpose the table
    receptions_slide_table = receptions_slide_table.transpose()
    receptions_slide_table.columns = receptions_slide_table.iloc[0]
    receptions_slide_table = receptions_slide_table[1:]

    # Save receptions slide table in the data/outputs folder
    receptions_slide_table.to_csv(
        f"{DATA_OUTPUTS_PATH}{RECEPTIONS_SLIDE_TABLE_NAME}",
        index=True,
    )


def create_radar_plot_inputs(
    normalised_player_receptions_metrics: pd.DataFrame,
    player_goals_metrics: pd.DataFrame,
    normalised_player_goals_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the inputs required for the radar plot. The radar plot inputs are created by merging the relevant metrics from the
    normalised player receptions metrics, player goals metrics, and normalised player goals metrics datasets. For
    non-normalised metrics to be used in the radar plot, the metrics are scaled to a 0-100 scale with 100 representing the
    maximum value of the metric across all players. This is to ensure the range of values for each metric is consistent across
    the radar plot. The radar plot inputs are returned in a dataset. The metrics used in the radar plot are:
    - Normalised shots and shot assists per 90 from receptions in zone 14 and 17
    - Normalised expected goals per 90 from receptions in zone 14 and 17
    - Normalised goals per 90
    - Scaled shots and shot assists per possession
    - Scaled ratio of goals to expected goals

    Args:
        normalised_player_receptions_metrics (pd.DataFrame): A dataset containing normalised reception metrics for each player
        player_goals_metrics (pd.DataFrame): A dataset containing goal metrics for each player
        normalised_player_goals_metrics (pd.DataFrame): A dataset containing normalised goal metrics for each player

    Returns:
        radar_plot_inputs (pd.DataFrame): A dataset containing the inputs required for the radar plot for each player
    """

    # Select and rename the relevant columns from the normalised player receptions metrics dataset
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

    # Select and rename the relevant columns from the normalised player goals metrics dataset
    normalised_player_goals_metrics = normalised_player_goals_metrics[
        [PLAYER_NAME, GOALS_PER_90]
    ].rename(columns={GOALS_PER_90: f"{NORMALISED}_{GOALS_PER_90}"})

    # Select the relevant columns from the player goals metrics dataset
    player_goals_metrics = player_goals_metrics[
        [
            PLAYER_NAME,
            SHOOTING_OPPORTUNITIES_PER_POSSESSION,
            RATIO_OF_GOALS_TO_EXPECTED_GOALS,
        ]
    ]

    # Determine the non-normalised metrics to be scaled from the player goals metrics dataset
    metrics_to_be_scaled = [
        metric for metric in player_goals_metrics.columns if metric not in [PLAYER_NAME]
    ]

    # Scale the non-normalised metrics to a 0-100 scale with 100 representing the maximum value of the metric across all
    # players
    for metric in metrics_to_be_scaled:
        player_goals_metrics[metric] = (
            player_goals_metrics[metric] * 100 / player_goals_metrics[metric].max()
        )

    # Merge the relevant metrics from the normalised player receptions metrics, player goals metrics, and normalised player
    # goals metrics datasets
    radar_plot_inputs = normalised_player_receptions_metrics.merge(
        normalised_player_goals_metrics, on=PLAYER_NAME, how=LEFT
    )
    radar_plot_inputs = radar_plot_inputs.merge(
        player_goals_metrics, on=PLAYER_NAME, how=LEFT
    )

    # Return the dataset containing the inputs required for the radar plot for each player
    return radar_plot_inputs


def create_box_plots_for_player_team_metrics(player_team_metrics, colour_mapping):
    """
    Create box plots for:
    - Opponent ELO rating by player
    - Difference in ELO rating of opponent from player's team by player

    Args:
        player_team_metrics (pd.DataFrame): A dataset containing match opponent metrics for each player
        colour_mapping (dict): A dictionary mapping each player to a colour
    """

    # Create box plot for opponent ELO rating by player
    create_box_plot(player_team_metrics, OPPOSITION_ELO, colour_mapping)

    # Create box plot for difference in ELO rating of opponent from player's team by player
    create_box_plot(player_team_metrics, ELO_DIFFERENCE, colour_mapping)


def create_scatter_plot_for_reception_metrics(
    player_receptions_metrics, normalised_player_receptions_metrics, colour_mapping
):
    """
    Create scatter plots for:
    - Shots and shot assists per 90 from receptions in zone 14 and 17 and expected goals per 90 from receptions in zone 14
    and 17
    - Normalised shots and shot assists per 90 from receptions in zone 14 and 17 and normalised expected goals per 90 from
    receptions in zone 14 and 17

    Args:
        player_receptions_metrics (pd.DataFrame): A dataset containing reception metrics for each player
        normalised_player_receptions_metrics (pd.DataFrame): A dataset containing normalised reception metrics for each player
        colour_mapping (dict): A dictionary mapping each player to a colour
    """

    # Create scatter plot for shots and shot assists per 90 from receptions in zone 14 and 17 and expected goals per 90 from
    # receptions in zone 14 and 17
    create_scatter_plot_of_two_metrics(
        player_receptions_metrics,
        SHOOTING_OPPORTUNITIES_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17,
        EXPECTED_GOALS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17,
        colour_mapping,
    )

    # Rename columns for normalised shots and shot assists per 90 from receptions in zone 14 and 17 and normalised expected goals
    # per 90 from receptions in zone 14 and 17
    normalised_player_receptions_metrics = normalised_player_receptions_metrics.rename(
        columns={
            SHOOTING_OPPORTUNITIES_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17: f"{NORMALISED}_{SHOOTING_OPPORTUNITIES_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17}",
            EXPECTED_GOALS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17: f"{NORMALISED}_{EXPECTED_GOALS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17}",
        }
    )

    # Create scatter plot for normalised shots and shot assists per 90 from receptions in zone 14 and 17 and normalised expected
    # goals per 90 from receptions in zone 14 and 17
    create_scatter_plot_of_two_metrics(
        normalised_player_receptions_metrics,
        f"{NORMALISED}_{SHOOTING_OPPORTUNITIES_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17}",
        f"{NORMALISED}_{EXPECTED_GOALS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17}",
        colour_mapping,
    )


def create_scatter_plot_for_goal_scoring_metrics(
    player_goals_metrics, normalised_player_goals_metrics, colour_mapping
):
    """
    Create scatter plots for:
    - Goals per 90 and expected goals per 90
    - Normalised goals per 90 and normalised expected goals per 90

    Args:
        player_goals_metrics (pd.DataFrame): A dataset containing goal scoring metrics for each player
        normalised_player_goals_metrics (pd.DataFrame): A dataset containing normalised goal scoring metrics for each player
        colour_mapping (dict): A dictionary mapping each player to a colour
    """

    # Create scatter plot for goals per 90 and expected goals per 90
    create_scatter_plot_of_two_metrics(
        player_goals_metrics,
        EXPECTED_GOALS_PER_90,
        GOALS_PER_90,
        colour_mapping,
        y_x_line=True,
    )

    # Rename columns for normalised goals per 90 and normalised expected goals per 90
    normalised_player_goals_metrics = normalised_player_goals_metrics.rename(
        columns={
            EXPECTED_GOALS_PER_90: f"{NORMALISED}_{EXPECTED_GOALS_PER_90}",
            GOALS_PER_90: f"{NORMALISED}_{GOALS_PER_90}",
        }
    )

    # Create scatter plot for normalised goals per 90 and normalised expected goals per 90
    create_scatter_plot_of_two_metrics(
        normalised_player_goals_metrics,
        f"{NORMALISED}_{EXPECTED_GOALS_PER_90}",
        f"{NORMALISED}_{GOALS_PER_90}",
        colour_mapping,
    )


def get_colour_palette_for_players(df):
    """
    Create a dictionary mapping each unique player in dataset to a seaborn husl colour

    Args:
        df (pd.DataFrame): A dataset containing player metrics

    Returns:
        player_color_mapping (dict): A dictionary mapping each unique player to a colour
    """

    # Get unique players in dataset
    unique_players = df[PLAYER_NAME].unique()

    # Create colour palette for number of unique players
    palette = sns.color_palette(HUSL, len(unique_players))

    # Create dictionary mapping each player to a colour
    player_color_mapping = {
        player: color for player, color in zip(unique_players, palette)
    }

    # Return dictionary
    return player_color_mapping


def analysis():
    """
    Perform analysis on the merged events and matches dataset to create visualisations for the final presentation. The
    analysis includes:
    - Creating box plots for opponent metrics by player
    - Plotting the proportion of receptions in each of the 18 pitch zones for each player
    - Creating a table to present key metrics for receptions
    - Creating scatter plots for reception metrics by player
    - Creating scatter plots for goal scoring metrics by player
    - Creating radar plot of key metrics by player
    """

    # Generate dataset showing each individual possession by each player
    possessions = possesions_dataset()

    # Generate dataset on opponent ELO rating and the difference in ELO rating of the opponent from the player's team
    # for matches faced by each player
    player_team_metrics = player_team_dataset()

    # Set colour palette for players
    player_color_mapping = get_colour_palette_for_players(player_team_metrics)

    # Create box plots for opponent ELO rating and the difference in ELO rating of the opponent from the player's team by
    # player
    create_box_plots_for_player_team_metrics(player_team_metrics, player_color_mapping)

    # Filter out only possessions that are from ball receipts
    receptions = possessions[possessions[TYPE_NAME] == BALL_RECEIPT]

    # Plot the proportion of receptions in each of the 18 pitch zones for each player
    plot_receptions_on_pitch(receptions)

    # Generate output metrics for receptions, before and after normalisation
    player_receptions_metrics = output_metrics_for_receptions(
        receptions, [PLAYER_ID, PLAYER_NAME]
    )
    normalised_player_receptions_metrics = output_normalised_metrics_for_receptions(
        receptions
    )

    # Create output table for key metrics for receptions
    create_receptions_slide_table(
        player_receptions_metrics, normalised_player_receptions_metrics
    )

    # Create scatter plots for receptions metrics
    create_scatter_plot_for_reception_metrics(
        player_receptions_metrics,
        normalised_player_receptions_metrics,
        player_color_mapping,
    )

    # Create output metrics for goals, before and after normalisation
    player_goals_metrics = output_metrics_for_goals(
        possessions, [PLAYER_ID, PLAYER_NAME]
    )
    normalised_player_goals_metrics = output_normalised_metrics_for_goals(possessions)

    # Create scatter plots for goal scoring metrics
    create_scatter_plot_for_goal_scoring_metrics(
        player_goals_metrics, normalised_player_goals_metrics, player_color_mapping
    )

    # Create radar plot of key metrics by player
    radar_plot_inputs = create_radar_plot_inputs(
        normalised_player_receptions_metrics,
        player_goals_metrics,
        normalised_player_goals_metrics,
    )
    create_radar_plot(radar_plot_inputs, player_color_mapping, CHART_LABELS)


if __name__ == "__main__":
    analysis()
