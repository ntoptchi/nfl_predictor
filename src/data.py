import pandas as pd
import numpy as np
from dateutil import parser
from .config import Config
import nfl_data_py as nfl

def load_games(seasons):
    # Schedule with results
    sched = nfl.import_schedules(seasons)
    # Standardize key fields
    sched =sched.rename(columns={
        "home_team": "home_team",
        "away_team": "away_team",
        "home_score": "home_score",
        "away_score": "away_score",
        "gameday": "gameday",
        "game_id": "game_id",
        "season": "season",
        "week": "week",
        "game_type": "game_type",
        "venue": "venue"
    })
    # Parse dates of games
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    # Filter regular season only
    sched = sched[sched["game_type"] == "REG"].copy()
    # Winner flag
    sched["home_win"] = (sched["home_score"] > sched["away_score"]).astype(int)
    sched["away_win"] = 1 - sched["home_win"]
    return sched

def team_game_stats(seasons):
    # Team game stats (offense/defense scores)
    # Using team game logs
    logs = nfl.import_team_game_logs(seasons)
    # Revelant cols
    cols = [
        "season", "week", "team", "opponent", "game_id",
        "game_date", "home_away",
        "points_for", "points_against",
        "passing_yards", "rushing_yards",
        "turnovers",
    ]
    # Some packed sources use different names; try fallback mapping
    remap = {
        "game_date": "game_date",
        "points_for": "points_for",
        "points_against": "points_against",
        "passing_yards": "passing_yards",
        "rushing_yards": "rushing_yards",
        "turnovers": "turnovers",
        "home_away": "home_away",
    }
    missing = [c for c in cols if c not in logs.columns]
    if missing:
        raise ValueError(f"Expected columns missing from team logs: {missing}")
    df = logs[cols].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df

def make_matchups(df, sched):
    # Flag home team
    df["home"] = (df["home_away"] == "HOME").astype(int)

    # Merge to ensure both teams on one row
    home = df[df["home"] == 1].copy()
    away = df[df["home"] == 0].copy()

    merged = home.merge(
        away,
        on=["game_id", "season", "week"],
        suffixes=("", "_opp"),
        validate="one_to_one"
    )

    # Rename for clarity
    merged = merged.rename(columns={
        "team": "home_team_code",
        "opponent": "home_opp_code",
        "team_opp": "away_team_code",
        "opponent_opp": "away_opp_code"
    })
    # Winner target from schedule (home win)
    sched_small = sched[["game_id", "home_win", "home_team", "away_team"]]
    out = merged.merge(sched_small, on="game_id", how="left")

    # Build a team-centric dataset by stacking home and away perspectives
    def select_cols(prefix):
        return {
            # team side
            f"{prefix}points_for": "points_for",
            f"{prefix}points_against": "points_against",
            f"{prefix}passing_yards": "passing_yards",
            f"{prefix}rushing_yards": "rushing_yards",
            f"{prefix}turnovers": "turnovers",
            f"{prefix}team_pf_roll": "team_pf_roll",
            f"{prefix}team_pa_roll": "team_pa_roll",
            f"{prefix}team_pass_y_roll": "team_pass_y_roll",
            f"{prefix}team_rush_y_roll": "team_rush_y_roll",
            f"{prefix}team_to_roll": "team_to_roll",
            f"{prefix}team_elo_momentum": "team_elo_momentum",
            f"{prefix}team_rest_days": "team_rest_days",
        }

    # Home perspective (team = home)
    home_cols = {
        "home": "home",
        **select_cols(""),
        **{k+"_opp": v.replace("team_", "opp_") for k, v in select_cols("").items()},
    }

    home_view = out.rename(columns=home_cols).copy()
    home_view["team_win"] = out["home_win"]

    # Away perspective (team = away)
    away_cols = {
        "home": "home",  # but will be 0 in away rows
        **{k+"_opp": v for k, v in select_cols("").items()},                # team side comes from away columns
        **{k: v.replace("team_", "opp_") for k, v in select_cols("").items()},  # opp side comes from home columns
    }
    away_view = out.rename(columns=away_cols).copy()
    away_view["team_win"] = 1 - out["home_win"]

    # Concatenate
    stacked = pd.concat([home_view, away_view], ignore_index=True)

    # Keep basic identifiers
    keep_id = ["game_id", "season", "week", "home_team", "away_team"]
    stacked = stacked.merge(
        sched[keep_id].drop_duplicates(),
        on="game_id", how="left"
    )

    return stacked

def build_dataset(seasons=None, rolling_n=None):
    seasons = seasons or Config.SEASONS
    rolling_n = rolling_n or Config.ROLLING_N

    sched = load_games(seasons)
    teamlogs = team_game_stats(seasons)
    teamlogs = add_rolling_features(teamlogs, rolling_n)
    dataset = make_matchups(teamlogs, sched)

    # Drop rows with missing target or critical rolls (first games can miss rolling)
    dataset = dataset.dropna(subset=["team_win"])
    # Some initial weeks have NaNs in rolling stats; drop or fill
    roll_cols = [c for c in dataset.columns if c.endswith("_roll")]
    dataset[roll_cols] = dataset[roll_cols].fillna(dataset[roll_cols].mean())

    return dataset