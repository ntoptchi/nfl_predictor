from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd

from .config import Config
from .data import build_dataset
from .model import load_model

def _select_best_side(df_with_probs: pd.DataFrame) -> pd.DataFrame:
    # keeps the higher-probability side per game
    return (
        df_with_probs.sort_values(["game_id", "team_prob_win"], ascending=[True, False])
                     .groupby("game_id", as_index=False)
                     .head(1)
                     .reset_index(drop=True)
    )

def _favorite_flag(s: pd.Series) -> pd.Series:
    # favorite if team spread is negative (requires merged market data)
    if "team_spread" not in s.index and "team_spread" not in s:
        return pd.Series([np.nan] * len(s))
    return (s["team_spread"] < 0).astype("Int64")

def _fmt_pct(x):
    return f"{100.0*x:.1f}%" if pd.notna(x) else ""

def evaluate(seasons: list[int], rolling_n: int, season_scope: list[int] | None = None):
    df = build_dataset(seasons=seasons, rolling_n=rolling_n)

    # If user wants a specific subset to report (e.g., a single season), filter AFTER features exist
    if season_scope:
        df = df[df["season"].isin(season_scope)].copy()
        if df.empty:
            raise SystemExit("No rows matched the requested season(s).")

    model = load_model()
    X = df[Config.FEATURES]
    df["team_prob_win"] = model.predict_proba(X)[:, 1]

    best = _select_best_side(df)

    # --- overall accuracy
    overall_acc = float((best["team_win"] == 1).mean())

    # --- accuracy by week
    wk_acc = (best.groupby("week")["team_win"].mean()
                    .rename("accuracy")
                    .reset_index()
                    .sort_values("week"))

    # --- home vs away (1 = home pick)
    ha_acc = (best.groupby("home")["team_win"].mean()
                    .rename("accuracy")
                    .reset_index()
                    .replace({"home": {0: "away_pick", 1: "home_pick"}}))

    # --- favorite vs underdog (requires team_spread)
    fav_tbl = pd.DataFrame()
    if "team_spread" in best.columns and best["team_spread"].notna().any():
        best["is_favorite_pick"] = (best["team_spread"] < 0).astype("Int64")
        fav_acc = (best.dropna(subset=["is_favorite_pick"])
                        .groupby("is_favorite_pick")["team_win"].mean()
                        .rename("accuracy").reset_index())
        fav_acc["bucket"] = fav_acc["is_favorite_pick"].map({1: "favorite_pick", 0: "underdog_pick"})
        fav_tbl = fav_acc[["bucket", "accuracy"]]
    else:
        fav_tbl = pd.DataFrame({"bucket": ["favorite_pick","underdog_pick"], "accuracy": [np.nan, np.nan]})

    return overall_acc, wk_acc, ha_acc, fav_tbl, best

def main():
    ap = ArgumentParser()
    ap.add_argument("--season", type=int, default=None, help="Evaluate a single season")
    ap.add_argument("--seasons", type=str, default=None, help="Range like 2015-2024")
    ap.add_argument("--rolling-n", type=int, default=Config.ROLLING_N)
    ap.add_argument("--out", type=str, default="reports/breakdowns.md")
    ap.add_argument("--csv", type=str, default="reports/breakdowns.csv", help="Flat CSV of weekly accuracy")
    args = ap.parse_args()

    # Determine build span (use your configured history unless overridden)
    if args.seasons:
        s, e = [int(x) for x in args.seasons.split("-")]
        build_span = list(range(s, e+1))
    else:
        build_span = Config.SEASONS

    # Reporting scope (if a single season provided)
    scope = [args.season] if args.season else None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    overall_acc, wk_acc, ha_acc, fav_tbl, best = evaluate(build_span, args.rolling_n, season_scope=scope)

    # --- Write Markdown report
    md = []
    title = f"Breakdowns — seasons {build_span[0]}–{build_span[-1]}" + (f" (reporting {scope[0]})" if scope else "")
    md.append(f"# {title}\n")
    md.append(f"**Overall accuracy:** {_fmt_pct(overall_acc)}\n")

    md.append("## Accuracy by Week\n")
    md.append(wk_acc.rename(columns={"week":"Week","accuracy":"Accuracy"})
              .assign(Accuracy=lambda d: d["Accuracy"].map(_fmt_pct))
              .to_markdown(index=False))
    md.append("")

    md.append("## Home vs. Away Picks\n")
    md.append(ha_acc.rename(columns={"home":"Pick side","accuracy":"Accuracy"})
              .assign(Accuracy=lambda d: d["Accuracy"].map(_fmt_pct))
              .to_markdown(index=False))
    md.append("")

    md.append("## Favorites vs. Underdogs (pick-based)\n")
    md.append(fav_tbl.rename(columns={"bucket":"Pick type","accuracy":"Accuracy"})
              .assign(Accuracy=lambda d: d["Accuracy"].map(lambda x: _fmt_pct(x) if pd.notna(x) else "N/A"))
              .to_markdown(index=False))
    md.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Wrote {args.out}")

    # --- Flat CSV (weekly)
    wk_acc.to_csv(args.csv, index=False)
    print(f"Wrote {args.csv}")

if __name__ == "__main__":
    main()
