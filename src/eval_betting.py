from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import Config
from .data import build_dataset, load_games
from .model import load_model

def american_to_decimal(odds):
    """Return decimal price from American odds (e.g., -150 -> 1.667, +150 -> 2.50)."""
    o = float(odds)
    return 1.0 + (100.0 / -o if o < 0 else o / 100.0)

def american_to_implied(odds):
    o = float(odds)
    return (100.0 / (o + 100.0)) if o > 0 else ((-o) / ((-o) + 100.0))

def settle_profit(stake, odds, win):
    """Profit for a single bet; stake returned separately by equity math:
       win: +stake*(decimal-1), lose: -stake"""
    dec = american_to_decimal(odds)
    return stake * (dec - 1.0) if win else -stake

def max_drawdown(equity):
    """Max drawdown of cumulative equity series."""
    arr = np.array(equity, dtype=float)
    peaks = np.maximum.accumulate(arr)
    drawdowns = (arr - peaks)
    return float(drawdowns.min())  # negative number

def pick_table_for_season(season: int, rolling_n: int):
    """Compute one pick per game (team + prob) for a season using the trained model."""
    # Build across all seasons up to 'season' so rolling features exist
    span = list(range(min(Config.SEASONS), season + 1))
    df = build_dataset(seasons=span, rolling_n=rolling_n)
    df_season = df[df["season"] == season].copy()
    if df_season.empty:
        raise SystemExit(f"No games found for season {season}.")

    model = load_model()
    df_season["team_prob_win"] = model.predict_proba(df_season[Config.FEATURES])[:, 1]

    # choose higher-probability side per game
    best = (df_season.sort_values(["game_id", "team_prob_win"], ascending=[True, False])
                      .groupby("game_id", as_index=False)
                      .head(1)
                      .reset_index(drop=True))

    # Attach home/away to know which moneyline to use
    sched = load_games([season])[["game_id","home_team","away_team","gameday","week","season"]]
    best = best.merge(sched, on="game_id", how="left")
    best["chosen_team"] = np.where(best["home"] == 1, best["home_team"], best["away_team"])
    best["is_home_pick"] = (best["home"] == 1).astype(int)
    best["won"] = (best["team_win"] == 1).astype(int)
    return best

def attach_market_moneylines(picks: pd.DataFrame, market_csv: str) -> pd.DataFrame:
    if not os.path.exists(market_csv):
        raise SystemExit(f"Market file not found: {market_csv}")
    mkt = pd.read_csv(market_csv)
    keep = ["game_id","ml_home","ml_away","close_spread_home","total_points"]
    mkt = mkt[[c for c in keep if c in mkt.columns]].drop_duplicates(subset=["game_id"], keep="last")
    out = picks.merge(mkt, on="game_id", how="left")
    # choose correct moneyline for the picked side
    out["ml_pick"] = np.where(out["is_home_pick"] == 1, out["ml_home"], out["ml_away"])
    return out

def simulate_bets(picks_mkt: pd.DataFrame, min_edge: float, kelly: float, flat: float):
    df = picks_mkt.copy()
    if "ml_pick" not in df.columns:
        raise SystemExit("No moneylines available; ensure market CSV includes ml_home/ml_away.")
    # implied probability from market
    df["p_imp"] = df["ml_pick"].apply(american_to_implied)
    df["edge"]  = df["team_prob_win"] - df["p_imp"]

    # keep only bets with edge >= threshold and valid moneyline
    df = df[df["ml_pick"].notna() & (df["edge"] >= min_edge)].copy()
    if df.empty:
        return df, 0.0, 0.0, 0.0, 0.0

    # stake sizing
    if kelly and kelly > 0:
        # Kelly for moneyline: b = decimal-1
        dec = df["ml_pick"].apply(american_to_decimal)
        b = dec - 1.0
        p = df["team_prob_win"]
        q = 1.0 - p
        f_star = (b * p - q) / b
        f_star = f_star.clip(lower=0)  # no negative bets
        df["stake"] = kelly * f_star
    elif flat and flat > 0:
        df["stake"] = float(flat)
    else:
        df["stake"] = 1.0  # default flat 1u if neither provided

    total_staked = float(df["stake"].sum())

    # settle
    df["profit"] = [settle_profit(stk, odds, win==1)
                    for stk, odds, win in zip(df["stake"], df["ml_pick"], df["won"])]

    equity = df["profit"].cumsum().tolist()
    roi = float(df["profit"].sum() / total_staked) if total_staked > 0 else 0.0
    mdd = max_drawdown(equity) if equity else 0.0

    return df, roi, total_staked, (float(df["won"].mean()) if len(df) else 0.0), mdd

def save_equity_curve(bets_df: pd.DataFrame, out_png: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    eq = bets_df["profit"].cumsum().values if not bets_df.empty else np.array([0.0])
    plt.figure(figsize=(7,4))
    plt.plot(eq)
    plt.xlabel("Bet #")
    plt.ylabel("Cumulative Profit (units)")
    plt.title("Equity Curve")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = ArgumentParser()
    ap.add_argument("--season", type=int, required=True, help="Season to simulate (uses that season's games)")
    ap.add_argument("--rolling-n", type=int, default=Config.ROLLING_N)
    ap.add_argument("--min-edge", type=float, default=0.02, help="Only bet when model edge >= threshold")
    ap.add_argument("--kelly", type=float, default=0.0, help="Fractional Kelly (0 to disable)")
    ap.add_argument("--flat", type=float, default=1.0, help="Flat stake units (ignored if kelly>0)")
    ap.add_argument("--market", type=str, default=os.path.join(Config.DATA_DIR, Config.FILE_MARKET))
    ap.add_argument("--out-csv", type=str, default="reports/bets.csv")
    ap.add_argument("--out-png", type=str, default="reports/equity_curve.png")
    args = ap.parse_args()

    os.makedirs("reports", exist_ok=True)

    picks = pick_table_for_season(args.season, args.rolling_n)
    picks_mkt = attach_market_moneylines(picks, args.market)

    bets, roi, staked, hit, mdd = simulate_bets(picks_mkt, args.min_edge, args.kelly, args.flat)

    # Save bets
    if not bets.empty:
        cols = ["season","week","gameday","game_id","home_team","away_team",
                "chosen_team","is_home_pick","team_prob_win","ml_pick","p_imp","edge","stake","won","profit"]
        bets_out = bets[[c for c in cols if c in bets.columns]].copy()
        bets_out.sort_values(["week","gameday"], inplace=True)
        bets_out.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv} ({len(bets_out)} bets)")
        save_equity_curve(bets_out, args.out_png)
        print(f"Wrote {args.out_png}")
    else:
        print("No qualifying bets after edge/market filters; nothing to save.")

    # Summary
    print(f"Total staked: {staked:.2f}u")
    print(f"ROI: {roi*100:.2f}%")
    print(f"Hit rate: {hit*100:.2f}%")
    print(f"Max drawdown: {mdd:.2f}u")

if __name__ == "__main__":
    main()
