import os, sys, pandas as pd, numpy as np
from typing import List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import Config
from src.data import build_dataset, load_games
from src.model import load_model
from src.eval_backtest import backtest as backtest_fn
from src.eval_betting import (pick_table_for_season, attach_market_moneylines,
                              simulate_bets)

def compute_week_picks(season: int, week: int):
    # Build until this season so rolling features exist
    all_seasons = list(range(min(Config.SEASONS), season + 1))
    df = build_dataset(seasons=all_seasons, rolling_n=Config.ROLLING_N)

    games = load_games([season])
    games = games[(games["season"] == season) & (games["week"] == week)][["game_id"]]
    df_wk = df.merge(games, on="game_id", how="inner").copy()
    if df_wk.empty:
        return pd.DataFrame()

    model = load_model()
    proba = model.predict_proba(df_wk[Config.FEATURES])[:, 1]
    df_wk["team_prob_win"] = proba

    best = (df_wk.sort_values(["game_id", "team_prob_win"], ascending=[True, False])
                 .groupby("game_id", as_index=False)
                 .head(1))
    # enrich with schedule fields
    sched = load_games([season])[["game_id","home_team","away_team","gameday","week","season"]]
    best = best.merge(sched, on="game_id", how="left")
    best["predicted_winner"] = np.where(best["home"] == 1, best["home_team"], best["away_team"])
    best["confidence"] = best["team_prob_win"]
    out = best[["game_id","season","week","gameday","home_team","away_team","predicted_winner","confidence"]].copy()
    out["gameday"] = out["gameday"].astype(str)
    return out

def split_top_and_flips(picks: pd.DataFrame, top_k: int = 5, flip_band: float = 0.02):
    if picks.empty:
        return picks, picks
    top = picks.sort_values("confidence", ascending=False).head(max(0, top_k)).copy()
    low, high = 0.5 - flip_band, 0.5 + flip_band
    flips = picks[(picks["confidence"] >= low) & (picks["confidence"] <= high)].copy()
    return top, flips

def run_backtest(s0: int, s1: int, rolling_n: int):
    span = list(range(s0, s1 + 1))
    return backtest_fn(span, rolling_n)

def run_betting_sim(season: int, min_edge: float, kelly: float, flat: float):
    picks = pick_table_for_season(season, Config.ROLLING_N)
    picks_mkt = attach_market_moneylines(picks, os.path.join(Config.DATA_DIR, Config.FILE_MARKET))
    return simulate_bets(picks_mkt, min_edge, kelly, flat)
