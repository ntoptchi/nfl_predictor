from argparse import ArgumentParser
import pandas as pd
from .data import build_dataset, load_games
from .model import load_model
from .config import Config

def week_matchups(season: int, week: int) -> pd.DataFrame:
    sched = load_games([season])
    wk = sched[(sched["season"] == season) & (sched["week"] == week)].copy()
    return wk[["game_id", "home_team", "away_team", "gameday", "week", "season"]]

def _pick_col(df: pd.DataFrame, base: str) -> str:
    """Return the first column name that exists among base, base_x, base_y."""
    for c in (base, f"{base}_x", f"{base}_y"):
        if c in df.columns:
            return c
    raise KeyError(f"Column '{base}' not found (nor with _x/_y) in columns: {list(df.columns)}")

def main():
    parser = ArgumentParser()
    parser.add_argument("--season", type=int, default=Config.CURRENT_SEASON)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--out", type=str, default=None, help="Optional CSV path for saving picks")
    parser.add_argument("--debug", action="store_true", help="Print shapes/columns for troubleshooting")
    args = parser.parse_args()

    # Build dataset through requested season so rolling features exist
    all_seasons = list(range(min(Config.SEASONS), args.season + 1))
    df = build_dataset(seasons=all_seasons, rolling_n=Config.ROLLING_N)

    # Filter to target week’s games
    games = week_matchups(args.season, args.week)
    if games.empty:
        print(f"No regular-season games found for season={args.season}, week={args.week}.")
        return

    df_wk = df.merge(games[["game_id"]], on="game_id", how="inner").copy()
    if df_wk.empty:
        print("Matched zero rows after merge — check that the dataset built successfully.")
        if args.debug:
            print("Dataset columns:", list(df.columns))
            print("Games columns:", list(games.columns))
        return

    if args.debug:
        print("df_wk shape:", df_wk.shape)
        print("df_wk columns (sample):", list(df_wk.columns)[:25])

    # Predict
    model = load_model()
    X = df_wk[Config.FEATURES]
    proba = model.predict_proba(X)[:, 1]  # prob that this 'team' row wins
    df_wk["team_prob_win"] = proba

    # Keep the higher-prob side for each game
    best = (
        df_wk.sort_values(["game_id", "team_prob_win"], ascending=[True, False])
            .groupby("game_id", as_index=False)
            .head(1)
    )

    # Merge back official labels (may create _x/_y)
    best = best.merge(games, on="game_id", how="left")

    # Resolve columns safely
    home_col   = _pick_col(best, "home_team")
    away_col   = _pick_col(best, "away_team")
    date_col   = _pick_col(best, "gameday")
    season_col = _pick_col(best, "season")
    week_col   = _pick_col(best, "week")

    # Winner based on which side the kept row represents
    best["predicted_winner"] = best.apply(
        lambda r: r[home_col] if r["home"] == 1 else r[away_col], axis=1
    )
    best["confidence"] = best["team_prob_win"]

    out = (
        best[[season_col, week_col, date_col, home_col, away_col, "predicted_winner", "confidence"]]
        .rename(columns={
            season_col: "season",
            week_col:   "week",
            date_col:   "gameday",
            home_col:   "home_team",
            away_col:   "away_team",
        })
        .sort_values(["gameday", "home_team"])
        .reset_index(drop=True)
    )

    # Print nicely
    pd.set_option("display.width", 120)
    print(out.to_string(index=False, justify="left", col_space=12, float_format=lambda x: f"{x:.3f}"))

    # Optional CSV
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\nSaved picks to {args.out}")

if __name__ == "__main__":
    main()
