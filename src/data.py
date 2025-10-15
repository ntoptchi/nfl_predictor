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
    """
    Build per-team, per-game stats from player weekly data (nfl_data_py 0.3.3).
    We aggregate passing yards, rushing yards, and turnovers (INT + fumbles lost).
    Points for/against are merged later from schedules for reliability.
    """

    wk = nfl.import_weekly_data(seasons)
    def pick(colnames, candidates, default=None):
        for c in candidates:
            if c in colnames:
                return c
        if default is not None:
            return default
        raise ValueError(f"None of the expected columns {candidates} found in weekly data")
    
    cols = set(wk.columns)

     # Helper that returns the first present name or None (never raises)
    def first_or_none(candidates):
        for c in candidates:
            if c in cols:
                return c
        return None


    team_col      = first_or_none(["recent_team", "team"])
    opp_col       = first_or_none(["opponent_team", "opponent"])
    week_col      = first_or_none(["week"])
    season_col    = first_or_none(["season"])
    game_id_col   = "game_id" if "game_id" in cols else None  
    home_away_col = first_or_none(["home_away"])

    pass_yds_col  = first_or_none(["pass_yds", "passing_yards"])
    rush_yds_col  = first_or_none(["rush_yds", "rushing_yards"])
    ints_col      = first_or_none(["interceptions", "int"])
    fumlost_col   = first_or_none(["fumbles_lost", "fum_lost"])


    required = [team_col, opp_col, week_col, season_col]
    if any(c is None for c in required):
        missing = ["team","opponent","week","season"]
        raise ValueError(f"Weekly data is missing required identifiers: {missing}. Found: {wk.columns.tolist()}")


    use_cols = [season_col, week_col, team_col, opp_col]
    if game_id_col:   use_cols.append(game_id_col)
    if home_away_col: use_cols.append(home_away_col)
    for c in [pass_yds_col, rush_yds_col, ints_col, fumlost_col]:
        if c: use_cols.append(c)

    df = wk[use_cols].copy()

    # Build turnovers (INT + fumbles lost); coerce missing to 0
    for c in [ints_col, fumlost_col, pass_yds_col, rush_yds_col]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["turnovers"]     = (df[ints_col] if ints_col in df else 0) + (df[fumlost_col] if fumlost_col in df else 0)
    df["passing_yards"] = df[pass_yds_col] if pass_yds_col in df else 0
    df["rushing_yards"] = df[rush_yds_col] if rush_yds_col in df else 0

    # If game_id/home_away missing, we’ll infer later from schedules during matchup join
    if game_id_col is None:
        df["game_id"] = np.nan
    else:
        df["game_id"] = df[game_id_col]

    if home_away_col is None:
        # infer later during make_matchups; placeholder
        df["home_away"] = np.nan
    else:
        df["home_away"] = df[home_away_col].str.upper()

    df.rename(columns={
        season_col: "season",
        week_col: "week",
        team_col: "team",
        opp_col: "opponent",
    }, inplace=True)

    # Aggregate player rows -> team-per-game totals
    group_keys = ["season", "week", "team", "opponent", "game_id", "home_away"]
    agg = (df.groupby(group_keys, dropna=False)[["passing_yards", "rushing_yards", "turnovers"]]
             .sum()
             .reset_index())

    # Attach a game_date to enable rest-day calcs later; we’ll merge from schedules in build_dataset
    return agg


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

     # If game_id/home_away missing from weekly data, infer via schedule
    if teamlogs["game_id"].isna().any() or teamlogs["home_away"].isna().any():
        # Home join
        home_side = sched[["game_id","season","week","home_team","away_team","gameday"]].copy()
        home_side["team"] = home_side["home_team"]
        home_side["opponent"] = home_side["away_team"]
        home_side["home_away"] = "HOME"

        # Away join
        away_side = sched[["game_id","season","week","home_team","away_team","gameday"]].copy()
        away_side["team"] = away_side["away_team"]
        away_side["opponent"] = away_side["home_team"]
        away_side["home_away"] = "AWAY"

        sched_team_rows = pd.concat([home_side, away_side], ignore_index=True)
        teamlogs = teamlogs.drop(columns=["game_id","home_away"], errors="ignore").merge(
            sched_team_rows[["game_id","season","week","team","opponent","home_away","gameday"]],
            on=["season","week","team","opponent"],
            how="left",
            validate="many_to_one"
        )
    else:
        # Ensure we have game_date as 'gameday' for rest calcs
        teamlogs = teamlogs.merge(
            sched[["game_id","gameday"]],
            on="game_id", how="left"
        )
    teamlogs.rename(columns={"gameday": "game_date"}, inplace=True)

    teamlogs = add_rolling_features(teamlogs, rolling_n)
    dataset = make_matchups(teamlogs, sched)

    # Drop rows with missing target or critical rolls (first games can miss rolling)
    dataset = dataset.dropna(subset=["team_win"])
    # Some initial weeks have NaNs in rolling stats; drop or fill
    roll_cols = [c for c in dataset.columns if c.endswith("_roll")]
    dataset[roll_cols] = dataset[roll_cols].fillna(dataset[roll_cols].mean())

    return dataset