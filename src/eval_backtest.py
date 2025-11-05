from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .config import Config
from .data import build_dataset
from .features import build_xy
from .model import build_ensemble

def _select_best_side(df_with_probs: pd.DataFrame) -> pd.DataFrame:
    # keeps the higher-probability side per game
    return (
        df_with_probs.sort_values(["game_id", "team_prob_win"], ascending=[True, False])
                     .groupby("game_id", as_index=False)
                     .head(1)
                     .reset_index(drop=True)
    )

def _brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))

def _season_folds(df: pd.DataFrame):
    seasons = sorted(df["season"].unique())
    for t in seasons[1:]:  # first season can't be tested (no prior train)
        train_idx = df.index[df["season"] < t].to_list()
        test_idx  = df.index[df["season"] == t].to_list()
        if len(train_idx) and len(test_idx):
            yield t, train_idx, test_idx

def backtest(seasons: list[int], rolling_n: int):
    # Build one dataset spanning all requested seasons (stable rolling features)
    all_df = build_dataset(seasons=seasons, rolling_n=rolling_n).reset_index(drop=True)
    X_all, y_all = build_xy(all_df)

    results = []
    for t, tr, te in _season_folds(all_df):
        model = build_ensemble()  # auto-load tuned params (from artifacts/best_params.json if present)
        model.fit(X_all.iloc[tr], y_all.iloc[tr])

        # predict per-row, then collapse to one side per game for scoring
        proba = model.predict_proba(X_all.iloc[te])[:, 1]
        test_df = all_df.iloc[te].copy()
        test_df["team_prob_win"] = proba

        best = _select_best_side(test_df)  # one row per game
        y_true = (best["team_win"] == 1).astype(int).to_numpy()
        y_prob = best["team_prob_win"].to_numpy()
        y_hat  = (y_prob >= 0.5).astype(int)

        acc   = float((y_hat == y_true).mean())
        ll    = float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6)))
        brier = _brier_score(y_true, y_prob)

        results.append({"season": t, "games": len(best), "accuracy": acc, "log_loss": ll, "brier": brier})

    res = pd.DataFrame(results).sort_values("season")
    if not res.empty:
        summary = {
            "season": "MEAN",
            "games": int(res["games"].sum()),
            "accuracy": float(res["accuracy"].mean()),
            "log_loss": float(res["log_loss"].mean()),
            "brier": float(res["brier"].mean()),
        }
        res = pd.concat([res, pd.DataFrame([summary])], ignore_index=True)
    return res

def main():
    ap = ArgumentParser()
    ap.add_argument("--seasons", type=str, default=None, help="Range like 2015-2024 (default: Config.SEASONS span)")
    ap.add_argument("--rolling-n", type=int, default=Config.ROLLING_N)
    ap.add_argument("--out-md", type=str, default="reports/backtest_summary.md")
    ap.add_argument("--out-csv", type=str, default="reports/backtest_metrics.csv")
    args = ap.parse_args()

    if args.seasons:
        s, e = [int(x) for x in args.seasons.split("-")]
        span = list(range(s, e + 1))
    else:
        span = Config.SEASONS

    os.makedirs("reports", exist_ok=True)
    res = backtest(span, args.rolling_n)

    # Save CSV
    res.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")

    # Save Markdown
    md = ["# Rolling-Origin Backtest",
          f"Seasons: {span[0]}–{span[-1]} | Rolling N={args.rolling_n}",
          "",
          res.rename(columns={"season": "Season", "games": "Games",
                              "accuracy": "Accuracy", "log_loss": "LogLoss", "brier": "Brier"}).to_markdown(index=False)]
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Wrote {args.out_md}")

if __name__ == "__main__":
    main()
