import pandas as pd
import numpy as np
from .config import Config
import nfl_data_py as nfl


# SCHEDULE 
def load_games(seasons):
    """
    Load schedule/results and standardize key fields.
    """
    sched = nfl.import_schedules(seasons)
    # Normalize expected columns
    sched = sched.rename(columns={
        "gameday": "gameday",
        "game_id": "game_id",
        "season": "season",
        "week": "week",
        "game_type": "game_type",
        "home_team": "home_team",
        "away_team": "away_team",
        "home_score": "home_score",
        "away_score": "away_score",
        "venue": "venue",
    })
    sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
    # Regular season only
    sched = sched[sched["game_type"] == "REG"].copy()
    # Winner flags
    sched["home_win"] = (sched["home_score"] > sched["away_score"]).astype(int)
    sched["away_win"] = 1 - sched["home_win"]
    return sched


# WEEKLY -> TEAM PER GAME 
def team_game_stats(seasons):
    """
    Build per-team, per-game totals from weekly player data (nfl-data-py 0.3.3).
    Aggregates passing_yards, rushing_yards, turnovers (INT + fumbles lost).
    'game_id' and 'home_away' may be missing here; we will infer from schedule later.
    """
    wk = nfl.import_weekly_data(seasons)
    cols = set(wk.columns)

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

    # Ensure we have the identifiers
    if any(c is None for c in [team_col, opp_col, week_col, season_col]):
        raise ValueError(f"Weekly data missing required identifiers. Found columns: {sorted(wk.columns)}")

    use_cols = [season_col, week_col, team_col, opp_col]
    if game_id_col:   use_cols.append(game_id_col)
    if home_away_col: use_cols.append(home_away_col)
    for c in [pass_yds_col, rush_yds_col, ints_col, fumlost_col]:
        if c: use_cols.append(c)

    df = wk[use_cols].copy()

    # Coerce numeric and fill missing with 0
    for c in [pass_yds_col, rush_yds_col, ints_col, fumlost_col]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["turnovers"]     = (df[ints_col] if ints_col in df else 0) + (df[fumlost_col] if fumlost_col in df else 0)
    df["passing_yards"] = df[pass_yds_col] if pass_yds_col in df else 0
    df["rushing_yards"] = df[rush_yds_col] if rush_yds_col in df else 0

    # Optional identifiers
    df["game_id"]   = df[game_id_col] if game_id_col else np.nan
    df["home_away"] = df[home_away_col].str.upper() if home_away_col else np.nan

    # Canonical names
    df.rename(columns={
        season_col: "season",
        week_col: "week",
        team_col: "team",
        opp_col: "opponent",
    }, inplace=True)

    # Aggregate player rows -> team-per-game
    group_keys = ["season", "week", "team", "opponent", "game_id", "home_away"]
    agg = (df.groupby(group_keys, dropna=False)[["passing_yards", "rushing_yards", "turnovers"]]
             .sum()
             .reset_index())

    return agg


# ROLLING FEATURES
def add_rolling_features(df, n):
    """
    Compute rolling means over last N games (per team), a simple momentum proxy,
    and rest days (time since previous game).
    Expects df to have: team, season, week, game_date, points_for, points_against, passing_yards, rushing_yards, turnovers
    """
    sort_cols = ["team", "season", "week"]
    df = df.sort_values(sort_cols).copy()
    grp = df.groupby("team", group_keys=False)

    # rolling means of team stats
    def roll(col, out):
        df[out] = grp[col].apply(lambda s: s.shift(1).rolling(n, min_periods=1).mean())

    roll("points_for",     "team_pf_roll")
    roll("points_against", "team_pa_roll")
    roll("passing_yards",  "team_pass_y_roll")
    roll("rushing_yards",  "team_rush_y_roll")
    roll("turnovers",      "team_to_roll")

    # simple momentum = scaled PF-PA
    df["team_elo_momentum"] = (df["team_pf_roll"] - df["team_pa_roll"]) / (df["team_pf_roll"].abs() + 1)

    # rest days
    df["prev_game_date"] = grp["game_date"].shift(1)
    df["team_rest_days"] = (df["game_date"] - df["prev_game_date"]).dt.days.fillna(10)
    df.drop(columns=["prev_game_date"], inplace=True)
    return df


# MATCHUPS
def make_matchups(df, sched):
    """
    Build stacked (team-vs-opp) rows so each game contributes two rows,
    one for the home perspective and one for the away perspective.
    """
    df["home"] = (df["home_away"] == "HOME").astype(int)

    # Defensive dedup: keep one row per side per game
    home = home.drop_duplicates(subset=["game_id"])  # one home row per game
    away = away.drop_duplicates(subset=["game_id"])  # one away row per game

    merged = home.merge(
        away,
        on=["game_id", "season", "week"],
        suffixes=("", "_opp"),
        validate="one_to_one"
    )

    # Winner target from schedule (home win)
    sched_small = sched[["game_id", "home_win", "home_team", "away_team"]]
    out = merged.merge(sched_small, on="game_id", how="left")

    # Home perspective (team = home)
    def select_cols(prefix):
        return {
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

    home_cols = {
        "home": "home",
        **select_cols(""),
        **{k + "_opp": v.replace("team_", "opp_") for k, v in select_cols("").items()},
    }
    home_view = out.rename(columns=home_cols).copy()
    home_view["team_win"] = out["home_win"]

    # Away perspective (team = away)
    away_cols = {
        "home": "home",  # will be 0 in these rows
        **{k + "_opp": v for k, v in select_cols("").items()},
        **{k: v.replace("team_", "opp_") for k, v in select_cols("").items()},
    }
    away_view = out.rename(columns=away_cols).copy()
    away_view["team_win"] = 1 - out["home_win"]

    stacked = pd.concat([home_view, away_view], ignore_index=True)

    # Keep identifiers
    keep_id = ["game_id", "season", "week", "home_team", "away_team"]
    stacked = stacked.merge(sched[keep_id].drop_duplicates(), on="game_id", how="left")
    return stacked


# FULL DATASET 
def build_dataset(seasons=None, rolling_n=None):
    seasons = seasons or Config.SEASONS
    rolling_n = rolling_n or Config.ROLLING_N

    # load schedule and weekly -> team-per-game aggregates
    sched = load_games(seasons)
    teamlogs = team_game_stats(seasons)

        # ---- Ensure one row per (game_id, team) ----
    num_cols = ["passing_yards", "rushing_yards", "turnovers"]
    take_first_cols = [
        "points_for", "points_against", "game_date", "home_away",
        "season", "week", "opponent"
    ]

    agg_spec = {c: "sum" for c in num_cols}
    agg_spec.update({c: "first" for c in take_first_cols})

    teamlogs = (teamlogs
                .sort_values(["season", "week", "team"])
                .groupby(["game_id", "team"], as_index=False)
                .agg(agg_spec))


    # Infer game_id, home_away, game_date, points_for/against from schedule
    home_side = sched[["game_id", "season", "week", "home_team", "away_team", "gameday", "home_score", "away_score"]].copy()
    home_side["team"] = home_side["home_team"]
    home_side["opponent"] = home_side["away_team"]
    home_side["home_away"] = "HOME"
    home_side["points_for"] = home_side["home_score"]
    home_side["points_against"] = home_side["away_score"]

    away_side = sched[["game_id", "season", "week", "home_team", "away_team", "gameday", "home_score", "away_score"]].copy()
    away_side["team"] = away_side["away_team"]
    away_side["opponent"] = away_side["home_team"]
    away_side["home_away"] = "AWAY"
    away_side["points_for"] = away_side["away_score"]
    away_side["points_against"] = away_side["home_score"]

    sched_team_rows = pd.concat([home_side, away_side], ignore_index=True)
    sched_team_rows.rename(columns={"gameday": "game_date"}, inplace=True)
    sched_team_rows["game_date"] = pd.to_datetime(sched_team_rows["game_date"], errors="coerce")

    # Merge schedule-derived identifiers/scores into teamlogs
    teamlogs = teamlogs.drop(columns=["game_id", "home_away"], errors="ignore").merge(
        sched_team_rows[
            ["game_id", "season", "week", "team", "opponent", "home_away", "game_date", "points_for", "points_against"]
        ],
        on=["season", "week", "team", "opponent"],
        how="left",
        validate="many_to_one"
    )

    # Rolling features & momentum/rest
    teamlogs = add_rolling_features(teamlogs, rolling_n)

    # Build team-vs-opp stacked dataset
    dataset = make_matchups(teamlogs, sched)

    # Drop rows with missing target (shouldn’t happen for completed games)
    dataset = dataset.dropna(subset=["team_win"])

    # Early-season rolling NaNs → fill with column means
    roll_cols = [c for c in dataset.columns if c.endswith("_roll")]
    if roll_cols:
        dataset[roll_cols] = dataset[roll_cols].fillna(dataset[roll_cols].mean())

    return dataset
