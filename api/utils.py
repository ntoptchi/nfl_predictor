from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# Safe imports / fallbacks

try:
    
    from src.data import build_dataset, load_games  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import build_dataset/load_games from src.data. "
        "Run uvicorn from project root (the folder that contains api/ and src/). "
        f"Original error: {e}"
    )


def _load_model_fallback() -> Any:
    """Fallback loader if src.data.load_model isn't available."""
    try:
        from src.config import Config  # type: ignore
    except Exception as e:
        raise ImportError(f"Missing src.config.Config (needed for model-path fallback). {e}")

    model_path = getattr(Config, "MODEL_PATH", None) or getattr(Config, "FILE_MODEL", None)
    if not model_path:
        raise RuntimeError(
            "No load_model() found and no Config.MODEL_PATH/FILE_MODEL set. "
            "Either add load_model() to src.data or set Config.MODEL_PATH."
        )

    import joblib
    return joblib.load(model_path)


try:
    from src.data import load_model  # type: ignore
except Exception:

    def load_model() -> Any:  # type: ignore
        return _load_model_fallback()


try:
    from src.config import Config  # type: ignore
except Exception as e:
    raise ImportError(f"Could not import Config from src.config. Original error: {e}")



# Picks computation

def compute_week_picks(season: int, week: int) -> pd.DataFrame:
    """
    One row per game with:
      game_id, season, week, gameday, home_team, away_team,
      home_win_prob, away_win_prob, predicted_winner, confidence

    Works even if build_dataset() does NOT include home_team/away_team/gameday.
    """
    base_seasons = list(getattr(Config, "SEASONS", []))
    if season not in base_seasons:
        base_seasons.append(season)
    all_seasons = sorted(set(base_seasons))

    df = build_dataset(seasons=all_seasons, rolling_n=getattr(Config, "ROLLING_N", 4))
    df_wk = df[(df["season"] == season) & (df["week"] == week)].copy()

    if df_wk.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "season",
                "week",
                "gameday",
                "home_team",
                "away_team",
                "home_win_prob",
                "away_win_prob",
                "predicted_winner",
                "confidence",
            ]
        )

    if "home" not in df_wk.columns:
        raise KeyError("compute_week_picks expected a 'home' column in the dataset.")

    # ---- Load schedule for this week (source of truth for home/away teams + date) ----
    sched = load_games(all_seasons)
    sched_wk = sched[(sched["season"] == season) & (sched["week"] == week)].copy()

    # Be defensive: support alternate column names
    col_home = "home_team" if "home_team" in sched_wk.columns else ("home" if "home" in sched_wk.columns else None)
    col_away = "away_team" if "away_team" in sched_wk.columns else ("away" if "away" in sched_wk.columns else None)
    col_day = "gameday" if "gameday" in sched_wk.columns else ("game_date" if "game_date" in sched_wk.columns else None)

    if col_home is None or col_away is None:
        raise KeyError(
            f"Schedule file is missing home/away team columns. Found columns: {list(sched_wk.columns)}"
        )

    keep_cols = ["game_id", "season", "week", col_home, col_away]
    if col_day is not None:
        keep_cols.append(col_day)

    sched_wk = sched_wk[keep_cols].drop_duplicates(subset=["game_id"]).copy()
    sched_wk = sched_wk.rename(
        columns={
            col_home: "home_team",
            col_away: "away_team",
            col_day: "gameday" if col_day else col_day,
        }
    )
    if "gameday" not in sched_wk.columns:
        sched_wk["gameday"] = ""

    # ---- Predict team-win probability for each stacked row ----
    model = load_model()
    X = df_wk[getattr(Config, "FEATURES")]
    df_wk["team_prob_win_raw"] = model.predict_proba(X)[:, 1]

    # ---- Get one home-row + one away-row per game_id ----
    home_rows = (
        df_wk[df_wk["home"] == 1]
        .sort_values(["game_id"])
        .drop_duplicates(subset=["game_id"], keep="first")[["game_id", "team_prob_win_raw"]]
        .rename(columns={"team_prob_win_raw": "home_prob_raw"})
    )

    away_rows = (
        df_wk[df_wk["home"] == 0]
        .sort_values(["game_id"])
        .drop_duplicates(subset=["game_id"], keep="first")[["game_id", "team_prob_win_raw"]]
        .rename(columns={"team_prob_win_raw": "away_prob_raw"})
    )

    games = home_rows.merge(away_rows, on="game_id", how="left").merge(
        sched_wk[["game_id", "season", "week", "gameday", "home_team", "away_team"]],
        on="game_id",
        how="left",
    )

    # ---- Normalize so home+away=1 (critical for non-identical percentages) ----
    denom = games["home_prob_raw"].fillna(0.0) + games["away_prob_raw"].fillna(0.0)
    denom = denom.replace(0, np.nan)

    games["home_win_prob"] = (games["home_prob_raw"] / denom).fillna(0.5)
    games["away_win_prob"] = (games["away_prob_raw"] / denom).fillna(0.5)

    games["predicted_winner"] = np.where(
        games["home_win_prob"] >= games["away_win_prob"],
        games["home_team"],
        games["away_team"],
    )
    games["confidence"] = games[["home_win_prob", "away_win_prob"]].max(axis=1)

    out = games[
        [
            "game_id",
            "season",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "home_win_prob",
            "away_win_prob",
            "predicted_winner",
            "confidence",
        ]
    ].copy()

    # JSON-safe
    out["game_id"] = out["game_id"].astype(str)
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out["gameday"] = out["gameday"].astype(str)
    out["home_team"] = out["home_team"].astype(str)
    out["away_team"] = out["away_team"].astype(str)
    out["predicted_winner"] = out["predicted_winner"].astype(str)
    out["home_win_prob"] = out["home_win_prob"].astype(float)
    out["away_win_prob"] = out["away_win_prob"].astype(float)
    out["confidence"] = out["confidence"].astype(float)

    return out



def _format_for_ui(picks_df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "matchup": f"{r['away_team']} @ {r['home_team']}",
            "pick": str(r["predicted_winner"]),
            "confidence": float(r["confidence"]),
        }
        for _, r in picks_df.iterrows()
    ]


def split_top_and_flips(
    picks_df: pd.DataFrame, top_k: int = 5, flip_band: float = 0.03
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (top_picks, coin_flips) in the exact dict format App.tsx expects.

    flip_band = distance from 0.50 considered a 'coin flip'
      e.g. 0.03 -> confidence in [0.47, 0.53]
    """
    if picks_df is None or picks_df.empty:
        return [], []

    df = picks_df.copy()
    df["flip_margin"] = (df["confidence"] - 0.5).abs()

    top_df = df.sort_values("confidence", ascending=False).head(int(top_k))
    flips_df = df[df["flip_margin"] <= float(flip_band)].sort_values("confidence", ascending=True)

    return _format_for_ui(top_df), _format_for_ui(flips_df)
