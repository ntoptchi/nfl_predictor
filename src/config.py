from dataclasses import dataclass

@dataclass
class Config:
    SEASONS = list(range(2015, 2025)) #inclusive start
    CURRENT_SEASON = 2025
    HOLDOUT_SEASON = 2024
    ROLLING_N = 5
    RANDOM_STATE = 42
    USE_XGBOOST = True
    MODEL_PATH = "artifacts/ensemble_model.joblib"
    PREP_PATH = "artifacts/prep.joblib" # scalers/encoders/etc
    CALIBRATE = True
    CALIBRATION_METHOD = "isotonic"
    USE_MARKET   = True
    USE_WEATHER  = True
    USE_QB       = True
    USE_EPA      = True
    FILE_MARKET  = "market_odds.csv"        # by game_id
    FILE_WEATHER = "weather.csv"            # by game_id
    FILE_QB      = "qb_status.csv"          # by season/week/team
    FILE_EPA     = "team_epa.csv"
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
    