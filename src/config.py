# Adjustable constants
PLAYER_IDS = list(range(5))
TEAM_IDS = list(range(6))

# File paths
DATA_RAW_PATH = "data/raw/"
DATA_PROCESSED_PATH = "data/processed/"
DATA_OUTPUTS_PATH = "data/outputs/"


# File names
EVENTS_DATA_NAME = "EventLevelData.csv"
MATCHES_DATA_NAME = "MatchLevelData.csv"
TRANSFOMED_EVENTS_DATA_NAME = "TransformedEventLevelData.csv"
RECEPTIONS_HEATMAP_NAME = "ReceptionsHeatmap.png"
RECEPTIONS_METRICS_NAME = "ReceptionsMetrics.csv"
PLAYER = "Player"
GOAL_METRICS_NAME = "GoalMetrics.csv"
TEAM = "Team"
TEAM_METRICS_NAME = "TeamMetrics.csv"
RECEPTIONS_SLIDE_TABLE_NAME = "ReceptionsSlideTable.csv"
SCATTER_PLOT_NAME = "ScatterPlot.png"


# Function constants
LEFT = "left"
SUM = "sum"
COUNT = "count"
STATISTIC = "statistic"
MEAN = "mean"
MEDIAN = "median"


# Plot constants
PITCH_TYPE = "statsbomb"
PITCH_COLOUR = "#03182F"
BLACK = "#000000"
WHITE = "#FFFFFF"
GREY = "#AFABAB"
BLUES = "Blues"
CENTER = "center"
TIGHT = "tight"
RIGHT = 'right'
BOTTOM = 'bottom'
COLOR = 'color'
TOP = 'top'
X_DIFF = 'x_diff'
Y_DIFF = 'y_diff'
HA = 'ha'
VA = 'va'
X_OFFSET = 'x_offset'
Y_OFFSET = 'y_offset'
BOTH = "both"
DASH_LINE = "--"
GOALS_EQUAL_XG = "Goals = xG"


# Events data constants
TYPE_NAME = "type_name"
EVENT_TIMESTAMP = "event_timestamp"
MATCH_PERIOD = "match_period"
MATCH_ID = "MatchID"
PLAYER_ID = "PlayerID"
END_OF_EVENT_LOCATION_X = "end_location_x"
END_OF_EVENT_LOCATION_Y = "end_location_y"
PRESSURE = "Pressure"
BALL_RECEIPT = "Ball Receipt*"
DUEL = "Duel"
LOCATION_X = "location_x"
LOCATION_Y = "location_y"
CARRY = "Carry"
DRIBBLE = "Dribble"
CARRY_END_LOCATION_X = "carry_end_location_x"
CARRY_END_LOCATION_Y = "carry_end_location_y"
DRIBBLE_END_LOCATION_X = "dribble_end_location_x"
DRIBBLE_END_LOCATION_Y = "dribble_end_location_y"
PASS = "Pass"
SHOT = "Shot"
CLEARANCE = "Clearance"
TYPE_RANK = "type_rank"
TYPE_ORDER = {
    BALL_RECEIPT: 1,
    DUEL: 2,
    PRESSURE: 3,
    DRIBBLE: 4,
    CARRY: 5,
    PASS: 6,
    SHOT: 7,
    CLEARANCE: 8,
}
PREV_END_LOCATION_X = "prev_end_location_x"
PREV_END_LOCATION_Y = "prev_end_location_y"
LOCATION_X_DIFF = "location_x_diff"
LOCATION_Y_DIFF = "location_y_diff"
PREV_TYPE_NAME = "prev_type_name"
PREV_DRIBBLE_OUTCOME = "prev_dribble_outcome"
DRIBBLE_OUTCOME_NAME = "dribble_outcome_name"
PREV_DUEL_OUTCOME = "prev_duel_outcome"
DUEL_OUTCOME_NAME = "duel_outcome_name"
START_OF_POSSESSION = "start_of_possession"
COMPLETE = "Complete"
WON = "Won"
POSSESSION_INDEX = "possession_index"
PREV_EVENT_TIMESTAMP = "prev_event_timestamp"
EVENT_TIMESTAMP_DIFF = "event_timestamp_diff"
PLAY_PATTERN_NAME = "play_pattern_name"
PREV_PLAY_PATTERN_NAME = "prev_play_pattern_name"
TEAM_ID = "TeamID"
TEAM_NAME = "TeamName"
PLAYER_NAME = "PlayerName"
GOAL = "goal"
PASS_ASSISTED_SHOT_ID = "pass_assisted_shot_id"
EXPECTED_GOALS = "expected_goals"
OUTCOME_TYPE_NAME = "outcome_type_name"
EVENT_VALUE = "EventValue"
MINUTES_PLAYED = "MinutesPlayed"
ZONE_14_AND_17 = "zone_14_and_17"
TOTAL_RECEPTIONS = "total_receptions"
TOTAL_RECEPTIONS_IN_ZONE_14_AND_17 = "total_receptions_in_zone_14_and_17"
SHOOTING_OPPORTUNITY = "shooting_opportunity"
TOTAL_SHOTS_IN_ZONE_14_AND_17 = "total_shots_in_zone_14_and_17"
TOTAL_SHOOTING_OPPORTUNITIES_IN_ZONE_14_AND_17 = "total_shooting_opportunities_in_zone_14_and_17"
PERCENTAGE_RECEPTIONS_IN_ZONE_14_AND_17 = "percentage_receptions_in_zone_14_and_17"
RECEPTIONS_PER_90 = "receptions_per_90"
RECEPTIONS_IN_ZONE_14_AND_17_PER_90 = "receptions_in_zone_14_and_17_per_90"
FINAL_THIRD = "final_third"
TOTAL_RECEPTIONS_IN_FINAL_THIRD = "total_receptions_in_final_third"
RECEPTIONS_IN_FINAL_THIRD_PER_90 = "receptions_in_final_third_per_90"
PERCENTAGE_RECEPTIONS_IN_FINAL_THIRD = "percentage_receptions_in_final_third"
SHOOTING_OPPORTUNITIES_PER_RECEPTION_IN_ZONE_14_AND_17 = "shooting_opportunities_per_reception_in_zone_14_and_17"
SHOOTING_OPPORTUNITIES_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17 = "shooting_opportunities_per_90_from_receptions_in_zone_14_and_17"
SHOTS_PER_RECEPTION_IN_ZONE_14_AND_17 = "shots_per_reception_in_zone_14_and_17"
SHOTS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17 = "shots_per_90_from_receptions_in_zone_14_and_17"
TOTAL_EXPECTED_GOALS_IN_ZONE_14_AND_17 = "total_expected_goals_in_zone_14_and_17"
EXPECTED_GOALS_PER_90_FROM_RECEPTIONS_IN_ZONE_14_AND_17 = "expected_goals_per_90_from_receptions_in_zone_14_and_17"
MEAN_EXPECTED_GOALS_PER_SHOT_FROM_RECEPTIONS_IN_ZONE_14_AND_17 = "mean_expected_goals_per_shot_from_receptions_in_zone_14_and_17"
MEDIAN_EXPECTED_GOALS_PER_SHOT_FROM_RECEPTIONS_IN_ZONE_14_AND_17 = "median_expected_goals_per_shot_from_receptions_in_zone_14_and_17"
MEAN_EVENT_SCORE_PER_RECEPTION_IN_ZONE_14_AND_17 = "mean_event_score_per_reception_in_zone_14_and_17"
MEDIAN_EVENT_SCORE_PER_RECEPTION_IN_ZONE_14_AND_17 = "median_event_score_per_reception_in_zone_14_and_17"
POSSESSION_EVENT_SCORE = "possession_event_score"
TOTAL_POSSESSIONS = "total_possessions"
TOTAL_SHOTS = "total_shots"
TOTAL_GOALS = "total_goals"
TOTAL_EXPECTED_GOALS = "total_expected_goals"
TOTAL_SHOOTING_OPPORTUNITIES = "total_shooting_opportunities"
SHOOTING_OPPORTUNITIES_PER_90 = "shooting_opportunities_per_90"
SHOOTING_OPPORTUNITIES_PER_POSSESSION = "shooting_opportunities_per_possession"
SHOTS_PER_90 = "shots_per_90"
SHOTS_PER_POSSESSION = "shots_per_possession"
GOALS_PER_90 = "goals_per_90"
GOALS_PER_SHOT = "goals_per_shot"
GOALS_PER_POSSESSION = "goals_per_possession"
EXPECTED_GOALS_PER_90 = "expected_goals_per_90"
MEAN_EXPECTED_GOALS_PER_SHOT = "mean_expected_goals_per_shot"
MEDIAN_EXPECTED_GOALS_PER_SHOT = "median_expected_goals_per_shot"
RATIO_OF_GOALS_TO_EXPECTED_GOALS = "ratio_of_goals_to_expected_goals"
PENALTY = "penalty"
PLAYER_TEAM_COMBO = 'player_team_combo'


# Events data chart labels
CHART_LABELS = {
    SHOOTING_OPPORTUNITIES_PER_POSSESSION: "Shots and shot assists per possession",
    SHOTS_PER_POSSESSION: "Shots per possession",
    SHOTS_PER_90: "Shots per 90",
    MEAN_EXPECTED_GOALS_PER_SHOT: "Mean xG per shot",
    EXPECTED_GOALS_PER_90: "xG per 90",
    GOALS_PER_90: "Goals per 90",    
}

# Matches data constants
HOME_TEAM_ID = "HomeTeamID"
AWAY_TEAM_ID = "AwayTeamID"
TEAM_ELO = "team_elo"
OPPOSITION_ELO = "opposition_elo"
TEAM_EXPECTED_GOALS = "team_expected_goals"
OPPONENT_EXPECTED_GOALS = "opponent_expected_goals"
HOME_TEAM_XG = "HomeTeamXG"
AWAY_TEAM_XG = "AwayTeamXG"
DIFFERENCE_TO_OPPONENT_ELO = "difference_to_opponent_elo"
DIFFERENCE_TO_OPPONENT_EXPECTED_GOALS = "difference_to_opponent_expected_goals"
MEAN_ELO_RATING = "mean_elo_rating"
MEAN_EXPECTED_GOALS = "mean_expected_goals"
HOME_ELO = "HomeElo"
AWAY_ELO = "AwayElo"
MEAN_OPPONENT_ELO_RATING = "mean_opponent_elo_rating"
