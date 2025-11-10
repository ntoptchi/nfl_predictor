import os
import sys
from typing import Tuple, List

import numpy as np
import pandas as pd

# Make sure we can import from src/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import Config
from src.data import build_dataset, load_games
from src.model import load_model
from src.eval_backtest import backtest as backtest_fn
from src.eval_betting import (
    pick_table_for_season,
    attach_market_moneylines,
    simulate_bets,
)


def compute_week_picks(season: int, week: int) -> pd.DataFrame:
    """
    Build features up to this season, filter to a specific week,
    run the model, and return one row per game with:
    game_id, season, week, gameday, home_team, away_team,
    predicted_winner, confidence
    """
    # Make sure we include all seasons used for training + the requested one
    base_seasons: List[int] = list(Config.SEASONS)
    if season not in base_seasons:
        base_seasons.append(season)
    all_seasons = sorted(set(base_seasons))

    # Build the full dataset (stacked team-side rows)
    df = build_dataset(seasons=all_seasons, rolling_n=Config.ROLLING_N)

    # Load schedule to know which games are in this week
    sched = load_games(all_seasons)
    games_week = sched[
        (sched["season"] == season) & (sched["week"] == week)
    ].copy()

    if games_week.empty:
        # No games found for this week
        return pd.DataFrame(
            columns=[
                "game_id",
                "season",
                "week",
                "gameday",
                "home_team",
                "away_team",
                "predicted_winner",
                "confidence",
            ]
        )

    # Keep only rows for games in this week
    df_wk = df.merge(
        games_week[["game_id"]],
        on="game_id",
        how="inner",
    ).copy()

    if df_wk.empty:
        # No feature rows matched; return empty, not an error
        return pd.DataFrame(
            columns=[
                "game_id",
                "season",
                "week",
                "gameday",
                "home_team",
                "away_team",
                "predicted_winner",
                "confidence",
            ]
        )

    # Load the trained ensemble model
    model = load_model()
    proba = model.predict_proba(df_wk[Config.FEATURES])[:, 1]
    df_wk["team_prob_win"] = proba

    # For each game, keep the side (home or away) with higher predicted win prob
    best = (
        df_wk.sort_values(["game_id", "team_prob_win"], ascending=[True, False])
        .groupby("game_id", as_index=False)
        .head(1)
    )

    # Attach home_team, away_team, gameday, week, season from the schedule
    sched_small = games_week[
        ["game_id", "home_team", "away_team", "gameday", "week", "season"]
    ].copy()
    best = best.merge(sched_small, on="game_id", how="left")

    # Now we are guaranteed to have home_team / away_team columns
    best["predicted_winner"] = np.where(
        best["home"] == 1,
        best["home_team"],
        best["away_team"],
    )
    best["confidence"] = best["team_prob_win"]

    out = best[
        [
            "game_id",
            "season",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "predicted_winner",
            "confidence",
        ]
    ].copy()
    out["gameday"] = out["gameday"].astype(str)
    return out


def split_top_and_flips(
    picks: pd.DataFrame, top_k: int = 5, flip_band: float = 0.02
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split full pick table into:
      - top_k highest confidence picks
      - "coin flips" where confidence is within flip_band of 0.5.
    """
    if picks.empty:
        return picks, picks

    top = (
        picks.sort_values("confidence", ascending=False)
        .head(max(0, top_k))
        .copy()
    )

    low, high = 0.5 - flip_band, 0.5 + flip_band
    flips = picks[
        (picks["confidence"] >= low) & (picks["confidence"] <= high)
    ].copy()

    return top, flips


def run_backtest(s0: int, s1: int, rolling_n: int) -> pd.DataFrame:
    """
    Wrapper around src.eval_backtest.backtest.
    Returns a DataFrame with columns like: season, games, accuracy, log_loss, brier.
    """
    span = list(range(s0, s1 + 1))
    return backtest_fn(span, rolling_n)


def run_betting_sim(
    season: int, min_edge: float, kelly: float, flat: float
):
    """
    Wrapper around src.eval_betting:
      - build pick table for a season
      - attach market moneylines
      - simulate bets
    Returns: bets_df, roi, total_staked, hit_rate, max_drawdown
    """
    picks = pick_table_for_season(season, Config.ROLLING_N)
    # paths pulled from Config so they match your project layout
    market_path = os.path.join(Config.DATA_DIR, Config.FILE_MARKET)
    picks_mkt = attach_market_moneylines(picks, market_path)
    return simulate_bets(picks_mkt, min_edge, kelly, flat)
