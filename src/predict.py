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
    df = df.copy()
    if "confidence" in df.columns:
        df["confidence"] = (df["confidence"] * 100).map(lambda x: f"{x:6.2f}%")
    cols = list(df.columns)
    col_widths = {c: max(len(c), *(len(str(v)) for v in df[c].astype(str))) for c in cols}
    header = "  ".join(f"{c:<{col_widths[c]}}" for c in cols)
    sep = "  ".join("-" * col_widths[c] for c in cols)
    lines = [header, sep]
    for _, row in df.iterrows():
        line = "  ".join(f"{str(row[c]):<{col_widths[c]}}" for c in cols)
        lines.append(line)
    return "\n".join(lines)

def _detect_out_format(path: str | None, forced: str | None) -> str:
    if forced: return forced
    if not path: return "console"
    p = path.lower()
    if p.endswith(".tsv"): return "tsv"
    if p.endswith(".md"):  return "md"
    if p.endswith(".txt"): return "txt"
    return "csv"

def main():
    parser = ArgumentParser()
    parser.add_argument("--season", type=int, default=Config.CURRENT_SEASON)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--out", type=str, default=None, help="Path to save output (extension decides default format)")
    parser.add_argument("--out-format", type=str, choices=["csv","tsv","md","txt"], default=None,
                        help="Force an output format: csv/tsv/md/txt (overrides extension)")
    # NEW: Top-confidence summary controls
    parser.add_argument("--top-k", type=int, default=5, help="How many safest picks to list")
    parser.add_argument("--flip-band", type=float, default=0.02,
                        help="Half-width around 0.5 for coin flips (e.g., 0.02 => 0.48–0.52)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Build dataset through requested season so rolling features exist
    all_seasons = list(range(min(Config.SEASONS), args.season + 1))
    df = build_dataset(seasons=all_seasons, rolling_n=Config.ROLLING_N)

    # Target week’s games
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
    ).merge(games, on="game_id", how="left")

    # Resolve columns safely
    home_col   = _pick_col(best, "home_team")
    away_col   = _pick_col(best, "away_team")
    date_col   = _pick_col(best, "gameday")
    season_col = _pick_col(best, "season")
    week_col   = _pick_col(best, "week")

    # Winner & confidence
    best["predicted_winner"] = best.apply(lambda r: r[home_col] if r["home"] == 1 else r[away_col], axis=1)
    best["confidence"] = best["team_prob_win"]

    # Final table
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

    # Console table
    pd.set_option("display.width", 120)
    print(out.to_string(index=False, justify="left", col_space=12, float_format=lambda x: f"{x:.3f}"))

    # --- NEW: Top-Confidence summary (console) ---
    # Top-K safest picks
    topk = out.sort_values("confidence", ascending=False).head(max(0, args.top_k)).copy()
    # Coin flips (within band around 0.5)
    low, high = 0.5 - args.flip_band, 0.5 + args.flip_band
    flips = out[(out["confidence"] >= low) & (out["confidence"] <= high)].copy() \
                .sort_values("gameday")

    def _fmt_pair(r):
        matchup = f"{r['away_team']} @ {r['home_team']}"
        confpct = f"{r['confidence']*100:0.2f}%"
        return matchup, r["predicted_winner"], confpct

    print("\n=== Top Confidence Picks ===")
    if topk.empty:
        print("(none)")
    else:
        for _, r in topk.iterrows():
            matchup, pick, conf = _fmt_pair(r)
            print(f"{matchup:<17}  →  {pick:<4}  ({conf})")

    print("\n=== Coin Flips (|p-0.5| ≤ {0:.2f}) ===".format(args.flip_band))
    if flips.empty:
        print("(none)")
    else:
        for _, r in flips.iterrows():
            matchup, pick, conf = _fmt_pair(r)
            print(f"{matchup:<17}  →  {pick:<4}  ({conf})")

    # Optional file output
    fmt = _detect_out_format(args.out, args.out_format)
    if args.out:
        if fmt == "csv":
            out.to_csv(args.out, index=False)
        elif fmt == "tsv":
            out.to_csv(args.out, index=False, sep="\t")
        elif fmt == "md":
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(out.to_markdown(index=False))
                # append summary in markdown
                f.write("\n\n### Top Confidence Picks\n\n")
                if topk.empty:
                    f.write("_none_\n")
                else:
                    md_top = topk[["away_team","home_team","predicted_winner","confidence"]].copy()
                    md_top["confidence"] = (md_top["confidence"]*100).map(lambda x: f"{x:0.2f}%")
                    f.write(md_top.rename(columns={
                        "away_team":"away",
                        "home_team":"home",
                        "predicted_winner":"pick"
                    }).to_markdown(index=False))
                f.write("\n\n### Coin Flips (|p−0.5| ≤ {0:.2f})\n\n".format(args.flip_band))
                if flips.empty:
                    f.write("_none_\n")
                else:
                    md_flip = flips[["away_team","home_team","predicted_winner","confidence"]].copy()
                    md_flip["confidence"] = (md_flip["confidence"]*100).map(lambda x: f"{x:0.2f}%")
                    f.write(md_flip.rename(columns={
                        "away_team":"away",
                        "home_team":"home",
                        "predicted_winner":"pick"
                    }).to_markdown(index=False))
        elif fmt == "txt":
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(_to_fixed_width(out))
                # append summary in text
                f.write("\n\n=== Top Confidence Picks ===\n")
                if topk.empty:
                    f.write("(none)\n")
                else:
                    for _, r in topk.iterrows():
                        matchup, pick, conf = _fmt_pair(r)
                        f.write(f"{matchup:<17}  →  {pick:<4}  ({conf})\n")
                f.write("\n=== Coin Flips (|p-0.5| ≤ {0:.2f}) ===\n".format(args.flip_band))
                if flips.empty:
                    f.write("(none)\n")
                else:
                    for _, r in flips.iterrows():
                        matchup, pick, conf = _fmt_pair(r)
                        f.write(f"{matchup:<17}  →  {pick:<4}  ({conf})\n")
        else:  # default csv
            out.to_csv(args.out, index=False)
        print(f"\nSaved picks to {args.out} ({fmt})")

if __name__ == "__main__":
    main()
