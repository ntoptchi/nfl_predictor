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
    FEATURES = [
        #rolling means
        "team_pf_roll", "team_pa_roll", "team_pass_y_roll", "team_rush_y_roll", "team_to_roll",
        "opp_pf_roll", "opp_pa_roll", "opp_pass_y_roll", "opp_rush_y_roll", "opp_to_roll",
        # context
            "team_rest_days", "opp_rest_days",
            "home",
            "team_elo_momentum", "opp_elo_momentum",
    ]
    TARGET = "team_win"
    