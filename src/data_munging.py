import pandas as pd
import numpy as np
from src.config import *


def read_and_transform_events_data() -> pd.DataFrame:
    """
    Read in raw events data and label events in the same possession before saving the output as a csv file in processed folder.
    Possessions are defined as a sequence of events where the ball is under the control of the same player. Events that start
    a new possession are identified if one of the following criteria are met, as long as the previous event was not a
    completed dribble or duel that was won:
        - The event is a ball receipt or pressure event
        - The x or y location of the event is more than 2 away from the previous event
        - The time between the current event and the previous event is more than 20 seconds
        - The play pattern changes from the previous event

    Returns:
        events_df (pd.DataFrame): Transformed events data with possession index attributed to each event
    """

    # Read in raw events data
    events_df = pd.read_csv(f"{DATA_RAW_PATH}{EVENTS_DATA_NAME}")

    # Sort data into event order per match and player, using match period, event time and event type (to ensure dribbles
    # and carries are in the correct order)
    events_df[TYPE_RANK] = events_df[TYPE_NAME].map(TYPE_ORDER)
    events_df = events_df.sort_values(
        by=[
            PLAYER_ID,
            MATCH_ID,
            MATCH_PERIOD,
            EVENT_TIMESTAMP,
            TYPE_RANK,
        ]
    )

    # Find previous end location for events, with the end location for ball receipts and duels being the same as the start and
    # the end location for carries being the carry end location
    events_df.loc[
        events_df[TYPE_NAME].isin([BALL_RECEIPT, DUEL]), END_OF_EVENT_LOCATION_X
    ] = events_df.loc[events_df[TYPE_NAME].isin([BALL_RECEIPT, DUEL]), LOCATION_X]
    events_df.loc[
        events_df[TYPE_NAME].isin([BALL_RECEIPT, DUEL]), END_OF_EVENT_LOCATION_Y
    ] = events_df.loc[events_df[TYPE_NAME].isin([BALL_RECEIPT, DUEL]), LOCATION_Y]
    events_df.loc[events_df[TYPE_NAME] == CARRY, END_OF_EVENT_LOCATION_X] = (
        events_df.loc[events_df[TYPE_NAME] == CARRY, CARRY_END_LOCATION_X]
    )
    events_df.loc[events_df[TYPE_NAME] == CARRY, END_OF_EVENT_LOCATION_Y] = (
        events_df.loc[events_df[TYPE_NAME] == CARRY, CARRY_END_LOCATION_Y]
    )

    # Calculate the absolute difference in x and y location between the current event and the previous event end location
    events_df[PREV_END_LOCATION_X] = events_df.groupby(
        [MATCH_ID, PLAYER_ID, MATCH_PERIOD]
    )[END_OF_EVENT_LOCATION_X].shift(1)
    events_df[PREV_END_LOCATION_Y] = events_df.groupby(
        [MATCH_ID, PLAYER_ID, MATCH_PERIOD]
    )[END_OF_EVENT_LOCATION_Y].shift(1)
    events_df[LOCATION_X_DIFF] = abs(
        events_df[LOCATION_X] - events_df[PREV_END_LOCATION_X]
    )
    events_df[LOCATION_Y_DIFF] = abs(
        events_df[LOCATION_Y] - events_df[PREV_END_LOCATION_Y]
    )

    # Find previous event type, dribble outcome, duel outcome, event timestamp and play pattern for each event, with the
    # event timestamp difference between the current event and the previous event calculated
    events_df[PREV_TYPE_NAME] = events_df.groupby([MATCH_ID, PLAYER_ID, MATCH_PERIOD])[
        TYPE_NAME
    ].shift(1)
    events_df[PREV_DRIBBLE_OUTCOME] = events_df.groupby(
        [MATCH_ID, PLAYER_ID, MATCH_PERIOD]
    )[DRIBBLE_OUTCOME_NAME].shift(1)
    events_df[PREV_DUEL_OUTCOME] = events_df.groupby(
        [MATCH_ID, PLAYER_ID, MATCH_PERIOD]
    )[DUEL_OUTCOME_NAME].shift(1)
    events_df[PREV_EVENT_TIMESTAMP] = events_df.groupby(
        [MATCH_ID, PLAYER_ID, MATCH_PERIOD]
    )[EVENT_TIMESTAMP].shift(1)
    events_df[EVENT_TIMESTAMP_DIFF] = (
        pd.to_datetime(events_df[EVENT_TIMESTAMP])
        - pd.to_datetime(events_df[PREV_EVENT_TIMESTAMP])
    ) / pd.Timedelta(seconds=1)
    events_df[PREV_PLAY_PATTERN_NAME] = events_df.groupby(
        [MATCH_ID, PLAYER_ID, MATCH_PERIOD]
    )[PLAY_PATTERN_NAME].shift(1)

    # Label the start of each possession within the same match, player and match period
    events_df[START_OF_POSSESSION] = False
    events_df.loc[
        (
            (events_df[LOCATION_X_DIFF] > 2)
            | (events_df[LOCATION_X_DIFF].isnull())
            | (events_df[LOCATION_Y_DIFF] > 2)
            | (events_df[LOCATION_Y_DIFF].isnull())
            | (events_df[TYPE_NAME].isin([BALL_RECEIPT, PRESSURE]))
            | (events_df[EVENT_TIMESTAMP_DIFF] > 20)
            | (events_df[PREV_PLAY_PATTERN_NAME] != events_df[PLAY_PATTERN_NAME])
        )
        & (events_df[PREV_DRIBBLE_OUTCOME] != COMPLETE)
        & (events_df[PREV_DUEL_OUTCOME] != WON),
        START_OF_POSSESSION,
    ] = True

    # Label events within the same possession with the same possession index
    events_df[START_OF_POSSESSION] = events_df[START_OF_POSSESSION].astype(int)
    events_df[POSSESSION_INDEX] = events_df[START_OF_POSSESSION].cumsum()

    # Drop unneccesary columns used for calculating possession index
    events_df = events_df.drop(
        columns=[
            TYPE_RANK,
            END_OF_EVENT_LOCATION_X,
            END_OF_EVENT_LOCATION_Y,
            PREV_END_LOCATION_X,
            PREV_END_LOCATION_Y,
            LOCATION_X_DIFF,
            LOCATION_Y_DIFF,
            PREV_TYPE_NAME,
            PREV_DRIBBLE_OUTCOME,
            PREV_DUEL_OUTCOME,
            PREV_EVENT_TIMESTAMP,
            EVENT_TIMESTAMP_DIFF,
            PREV_PLAY_PATTERN_NAME,
        ]
    )

    # Save transformed events data
    events_df.to_csv(f"{DATA_PROCESSED_PATH}{TRANSFOMED_EVENTS_DATA_NAME}", index=False)

    # Return transformed events data
    return events_df


def sigmoid(x: float, k: float, x0: float) -> float:
    """
    Calculate output of sigmoid function for given input x, k and x0.

    Args:
        x (float): Input value
        k (float): Sigmoid function slope
        x0 (float): Sigmoid function midpoint

    Returns:
        output (float): Output of sigmoid function
    """

    # Calculate output of sigmoid function
    output = 1 / (1 + np.exp(-k * (x - x0)))

    # Return output of sigmoid function
    return output


def create_opponent_elo_weighting(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """
    Create opponent ELO weighting for each event in the dataset, with the weighting calculated using a sigmoid function.
    The sigmoid function is centred around the median opponent ELO rating, with the slope of the function determined by the range
    of opponent ELO ratings in the dataset. The weighting is capped at a chosen minimum value.

    Args:
        df (pd.DataFrame): DataFrame containing events data with opponent ELO ratings merged from matches data
        cap (float): Minimum value for opponent ELO weighting

    Returns:
        df (pd.DataFrame): DataFrame containing events data with opponent ELO weighting column added
    """

    # Find median, maximum and minimum opponent ELO rating across all opponents faced in matches played in by targets. Events
    # data has repeated rows for same match so first need to group by match and team to get single opponent ELO rating for
    # each match
    opposition_elo_df = df.groupby([MATCH_ID, TEAM_ID], as_index=False, dropna=False)[
        OPPOSITION_ELO
    ].max()
    opposition_elo_median = opposition_elo_df[OPPOSITION_ELO].median()
    opposition_elo_max = opposition_elo_df[OPPOSITION_ELO].max()
    opposition_elo_min = opposition_elo_df[OPPOSITION_ELO].min()

    # Calculate the slope of sigmoid function for opponent ELO weighting using the range of opponent ELO ratings, with the
    # SIGMOID_K_RANGE chosen to limit the range of possible z values and therefore min and max values of sigmoid function
    k = SIGMOID_K_RANGE / (opposition_elo_max - opposition_elo_min)

    # Set midpoint of sigmoid function to be the median opponent ELO rating
    x0 = opposition_elo_median

    # Calculate opponent ELO weighting for each event using sigmoid function
    df[OPPOSITION_ELO_WEIGHTING] = sigmoid(df[OPPOSITION_ELO], k, x0)

    # Cap opponent ELO weighting at chosen minimum value
    df[OPPOSITION_ELO_WEIGHTING] = df[OPPOSITION_ELO_WEIGHTING].apply(
        lambda x: max(x, cap)
    )

    # Return DataFrame with opponent ELO weighting column added
    return df


def create_elo_difference_weighting(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """
    Create ELO difference weighting for each event in the dataset, with the weighting calculated using a sigmoid function.
    The ELO difference is calculated as the difference of the opponent ELO rating from the player's team ELO rating. The
    sigmoid function is centred around 0, with the slope of the function determined by the range of ELO differences in the
    dataset. The weighting is capped at a chosen minimum value.

    Args:
        df (pd.DataFrame): DataFrame containing events data with team and opponent ELO ratings merged from matches data
        cap (float): Minimum value for ELO difference weighting

    Returns:
        df (pd.DataFrame): DataFrame containing events data with ELO difference weighting column added
    """

    # Calculate ELO difference between player's team and opponent for each event
    df[ELO_DIFFERENCE] = df[OPPOSITION_ELO] - df[TEAM_ELO]

    # Find maximum and minimum ELO difference across all opponents faced in matches played in by targets
    elo_difference_max = df[ELO_DIFFERENCE].max()
    elo_difference_min = df[ELO_DIFFERENCE].min()

    # Calculate the slope of sigmoid function for ELO difference weighting using the range of ELO differences, with the
    # SIGMOID_K_RANGE chosen to limit the range of possible z values and therefore min and max values of sigmoid function
    k = SIGMOID_K_RANGE / (elo_difference_max - elo_difference_min)

    # Set midpoint of sigmoid function to be 0
    x0 = 0

    # Calculate ELO difference weighting for each event using sigmoid function
    df[ELO_DIFFERENCE_WEIGHTING] = sigmoid(df[ELO_DIFFERENCE], k, x0)

    # Cap ELO difference weighting at chosen minimum value
    df[ELO_DIFFERENCE_WEIGHTING] = df[ELO_DIFFERENCE_WEIGHTING].apply(
        lambda x: max(x, cap)
    )

    # Return DataFrame with ELO difference weighting column added
    return df


def transform_joined_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detrmine the player's team and opposition ELO ratings for each event in the dataset using the home and away team ELO
    ratings from the matches data. Create opponent ELO weighting and ELO difference weighting for each event in the dataset
    using sigmoid functions.

    Args:
        df (pd.DataFrame): DataFrame containing events data with matches data merged

    Returns:
        df (pd.DataFrame): DataFrame containing events data with team and opposition ELO ratings, opponent ELO weighting and
        ELO difference weighting columns added
    """

    # Determine player's team and opposition ELO ratings for each event in the dataset from the home and away team ELO ratings
    # in the matches data
    df.loc[df[TEAM_ID] == df[HOME_TEAM_ID], TEAM_ELO] = df.loc[
        df[TEAM_ID] == df[HOME_TEAM_ID], HOME_ELO
    ]
    df.loc[df[TEAM_ID] == df[AWAY_TEAM_ID], TEAM_ELO] = df.loc[
        df[TEAM_ID] == df[AWAY_TEAM_ID], AWAY_ELO
    ]
    df.loc[df[TEAM_ID] == df[HOME_TEAM_ID], OPPOSITION_ELO] = df.loc[
        df[TEAM_ID] == df[HOME_TEAM_ID], AWAY_ELO
    ]
    df.loc[df[TEAM_ID] == df[AWAY_TEAM_ID], OPPOSITION_ELO] = df.loc[
        df[TEAM_ID] == df[AWAY_TEAM_ID], HOME_ELO
    ]

    # Drop unneccesary matches columns from DataFrame
    df = df.drop(
        [
            HOME_TEAM_ID,
            AWAY_TEAM_ID,
            HOME_ELO,
            AWAY_ELO,
            HOME_TEAM_XG,
            AWAY_TEAM_XG,
            HOME_GOALS,
            AWAY_GOALS,
            MATCH_DATE_TIME,
        ],
        axis=1,
    )

    # Create opponent ELO weighting and ELO difference weighting for each event in the dataset using sigmoid functions
    df = create_opponent_elo_weighting(df, OPPOSITION_ELO_CAP)
    df = create_elo_difference_weighting(df, ELO_DIFFERENCE_CAP)

    # Return transformed joined data
    return df


def read_and_join_events_and_matches_data() -> pd.DataFrame:
    """
    Read in events and matches data, join the two datasets and transform the joined data before saving the output as a csv file
    in processed folder.

    Returns:
        df (pd.DataFrame): Joined and transformed events and matches data
    """

    # Read in events data and transform it to label events in the same possession
    events_df = read_and_transform_events_data()

    # Read in matches data
    matches_df = pd.read_csv(
        f"{DATA_RAW_PATH}{MATCH_LEVEL_DATA_CORRECTED_MAPPING_NAME}"
    )

    # Join matches data to transformed events data
    df = events_df.merge(matches_df, how=LEFT, on=MATCH_ID)

    # Transform joined data to determine player's team and opposition ELO ratings for each event in the dataset and create
    # opponent ELO weighting and ELO difference weighting for each event
    df = transform_joined_data(df)

    # Save joined data
    df.to_csv(
        f"{DATA_PROCESSED_PATH}{MERGED_EVENTS_MATCH_LEVEL_DATA_NAME}", index=False
    )

    # Return joined and transformed events and matches data
    return df


if __name__ == "__main__":
    read_and_join_events_and_matches_data()
