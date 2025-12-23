from dataclasses import dataclass
import os
@dataclass
class Config:
    SEASONS = list(range(2018, 2025)) #inclusive start
    CURRENT_SEASON = 2025
    HOLDOUT_SEASON = 2024
    ROLLING_N = 5
    RANDOM_STATE = 42
    USE_XGBOOST = True
    MODEL_PATH = "artifacts/ensemble_model.joblib"
    PREP_PATH = "artifacts/prep.joblib" # scalers/encoders/etc
    CALIBRATE = True
    CALIBRATION_METHOD = "isotonic"
    DATA_DIR = "data"
    USE_MARKET   = True
    USE_WEATHER  = True
    USE_QB       = True
    USE_EPA      = True
    FILE_MARKET  = "market_odds.csv"        # by game_id
    FILE_WEATHER = "weather.csv"            # by game_id
    FILE_QB      = "qb_status.csv"          # by season/week/team
    FILE_EPA     = "team_epa.csv"
    TUNE_TRIALS = 40            # try 40–200 depending on patience
    TUNE_TIMEOUT = None         # seconds; set to e.g. 1800 to cap
    TUNE_MIN_SEASONS = 5        # guard against tiny datasets
    BEST_PARAMS_PATH = "artifacts/best_params.json"

    ODDS_API_KEY    = os.getenv("ODDS_API_KEY",    "") #put your API key here
   

    FEATURES = [
        #rolling means
        "team_pf_roll", "team_pa_roll", "team_pass_y_roll", "team_rush_y_roll", "team_to_roll",
        "opp_pf_roll", "opp_pa_roll", "opp_pass_y_roll", "opp_rush_y_roll", "opp_to_roll",
        # context
            "team_rest_days", "opp_rest_days",
            "home",
            "team_elo_momentum", "opp_elo_momentum",
            "team_elo", "opp_elo","elo_diff",
            "team_spread", "opp_spread",              # ATS perspective (negative = favored)
            "team_ml_implied", "opp_ml_implied",      # moneyline implied prob (vig-raw)
            "total_points",                           # closing total (same for both sides)
            "wind_mph", "temp_f", "is_precip", "is_outdoor",
            "qb_changed", "starters_out",
            "off_epa_roll","def_epa_roll","sr_off_roll","sr_def_roll",
            "opp_off_epa_roll","opp_def_epa_roll","opp_sr_off_roll","opp_sr_def_roll",
    ]
    TARGET = "team_win"
    