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
    for c in (base, f"{base}_x", f"{base}_y"):
        if c in df.columns:
            return c
    raise KeyError(f"Column '{base}' not found (nor with _x/_y). Columns: {list(df.columns)}")

def _to_fixed_width(df: pd.DataFrame) -> str:
    """Return a pretty fixed-width text table."""
    # format confidence as % for readability in text
    df = df.copy()
    if "confidence" in df.columns:
        df["confidence"] = (df["confidence"] * 100).map(lambda x: f"{x:6.2f}%")

    cols = list(df.columns)
    # compute widths
    col_widths = {c: max(len(c), *(len(str(v)) for v in df[c].astype(str))) for c in cols}
    # header
    header = "  ".join(f"{c:<{col_widths[c]}}" for c in cols)
    sep = "  ".join("-" * col_widths[c] for c in cols)
    # rows
    lines = [header, sep]
    for _, row in df.iterrows():
        line = "  ".join(f"{str(row[c]):<{col_widths[c]}}" for c in cols)
        lines.append(line)
    return "\n".join(lines)

def main():
    parser = ArgumentParser()
    parser.add_argument("--season", type=int, default=Config.CURRENT_SEASON)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--out", type=str, default=None, help="Path to save output (extension decides default format)")
    parser.add_argument("--out-format", type=str, choices=["csv","tsv","md","txt"], default=None,
                        help="Force an output format: csv/tsv/md/txt (overrides extension)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    all_seasons = list(range(min(Config.SEASONS), args.season + 1))
    df = build_dataset(seasons=all_seasons, rolling_n=Config.ROLLING_N)

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

    model = load_model()
    X = df_wk[Config.FEATURES]
    proba = model.predict_proba(X)[:, 1]
    df_wk["team_prob_win"] = proba

    best = (
        df_wk.sort_values(["game_id", "team_prob_win"], ascending=[True, False])
            .groupby("game_id", as_index=False)
            .head(1)
    ).merge(games, on="game_id", how="left")

    home_col   = _pick_col(best, "home_team")
    away_col   = _pick_col(best, "away_team")
    date_col   = _pick_col(best, "gameday")
    season_col = _pick_col(best, "season")
    week_col   = _pick_col(best, "week")

    best["predicted_winner"] = best.apply(lambda r: r[home_col] if r["home"] == 1 else r[away_col], axis=1)
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

    # Always pretty-print to console
    pd.set_option("display.width", 120)
    print(out.to_string(index=False, justify="left", col_space=12, float_format=lambda x: f"{x:.3f}"))

    # Optional file output
    if args.out:
        fmt = (args.out_format or
               ("tsv" if args.out.lower().endswith(".tsv") else
                "md" if args.out.lower().endswith(".md") else
                "txt" if args.out.lower().endswith(".txt") else
                "csv"))

        if fmt == "csv":
            out.to_csv(args.out, index=False)
        elif fmt == "tsv":
            out.to_csv(args.out, index=False, sep="\t")
        elif fmt == "md":
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(out.to_markdown(index=False))
        elif fmt == "txt":
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(_to_fixed_width(out))
        print(f"\nSaved picks to {args.out} ({fmt})")

if __name__ == "__main__":
    main()
