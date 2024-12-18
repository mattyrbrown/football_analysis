import pandas as pd
import numpy as np
from src.config import *


def read_and_transform_events_data():

    # Read in events data
    events_df = pd.read_csv(f"{DATA_RAW_PATH}{EVENTS_DATA_NAME}")

    # Sort data into event order per match and player
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

    # Find previous end location for events, where neccesary, and previous event types and outcomes
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
        pd.to_datetime(events_df[EVENT_TIMESTAMP]) - pd.to_datetime(events_df[PREV_EVENT_TIMESTAMP])
    ) / pd.Timedelta(seconds=1)
    events_df[PREV_PLAY_PATTERN_NAME] = events_df.groupby(
        [MATCH_ID, PLAYER_ID, MATCH_PERIOD]
    )[PLAY_PATTERN_NAME].shift(1)

    # Identify the start of each possession
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

    # Label events within the same possession
    events_df[START_OF_POSSESSION] = events_df[START_OF_POSSESSION].astype(int)
    events_df[POSSESSION_INDEX] = events_df[START_OF_POSSESSION].cumsum()

    # Drop unneccesary columns
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


def read_and_join_events_and_matches_data():
    
    # Read and transform events data
    events_df = read_and_transform_events_data()

    # Read in matches data
    matches_df = pd.read_csv(f"{DATA_RAW_PATH}{MATCHES_DATA_NAME}")

    # Join matches data to events data
    df = events_df.merge(
        matches_df,
        how=LEFT,
        on=MATCH_ID
    )

    return df


if __name__ == "__main__":
    read_and_transform_events_data()
