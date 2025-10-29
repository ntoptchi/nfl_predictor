from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import Config
from .data import build_dataset
from .model import load_model

def _select_best_side(df_with_probs: pd.DataFrame) -> pd.DataFrame:
    return (
        df_with_probs.sort_values(["game_id", "team_prob_win"], ascending=[True, False])
                     .groupby("game_id", as_index=False)
                     .head(1)
                     .reset_index(drop=True)
    )

def _calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = pd.cut(y_prob, edges, include_lowest=True)
    df = pd.DataFrame({"bin": bins, "y_prob": y_prob, "y_true": y_true})
    grouped = df.groupby("bin", observed=True).agg(
        count=("y_true", "size"),
        mean_pred=("y_prob", "mean"),
        mean_true=("y_true", "mean"),
    ).reset_index()
    # add bin center for plotting
    grouped["bin_center"] = grouped["bin"].apply(lambda x: (x.left + x.right) / 2 if pd.notna(x) else np.nan)
    return grouped

def _brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))

def main():
    ap = ArgumentParser()
    ap.add_argument("--season", type=int, default=None, help="Evaluate a single season (common)")
    ap.add_argument("--seasons", type=str, default=None, help="Range like 2015-2024 (optional)")
    ap.add_argument("--rolling-n", type=int, default=Config.ROLLING_N)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="reports")
    args = ap.parse_args()

    if args.seasons:
        s, e = [int(x) for x in args.seasons.split("-")]
        build_span = list(range(s, e+1))
    else:
        build_span = Config.SEASONS

    # Build dataset across history, then filter for reporting season if provided
    df = build_dataset(seasons=build_span, rolling_n=args.rolling_n)
    if args.season:
        df = df[df["season"] == args.season].copy()
        if df.empty:
            raise SystemExit(f"No rows found for season {args.season}.")

    os.makedirs(args.outdir, exist_ok=True)

    model = load_model()
    X = df[Config.FEATURES]
    df["team_prob_win"] = model.predict_proba(X)[:, 1]

    best = _select_best_side(df)
    y_true = (best["team_win"] == 1).astype(int).to_numpy()
    y_prob = best["team_prob_win"].to_numpy()

    # --- metrics
    brier = _brier_score(y_true, y_prob)
    bins_df = _calibration_bins(y_true, y_prob, n_bins=args.bins)

    # --- plots
    # Reliability curve
    fig1 = plt.figure(figsize=(6, 6))
    ax1 = fig1.add_subplot(111)
    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax1.plot(bins_df["mean_pred"], bins_df["mean_true"], marker="o")
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Empirical win rate")
    title = f"Calibration (Reliability) — Brier={brier:.3f}"
    if args.season: title += f" — Season {args.season}"
    ax1.set_title(title)
    fig1.tight_layout()
    cal_path = os.path.join(args.outdir, "calibration_curve.png")
    fig1.savefig(cal_path, dpi=160)
    plt.close(fig1)

    # Histogram
    fig2 = plt.figure(figsize=(6, 4))
    ax2 = fig2.add_subplot(111)
    ax2.hist(y_prob, bins=20, edgecolor="black")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Probability Histogram" + (f" — Season {args.season}" if args.season else ""))
    fig2.tight_layout()
    hist_path = os.path.join(args.outdir, "prob_hist.png")
    fig2.savefig(hist_path, dpi=160)
    plt.close(fig2)

    # Save bins CSV
    bins_csv = os.path.join(args.outdir, "calibration_bins.csv")
    bins_df.to_csv(bins_csv, index=False)

    print(f"Saved: {cal_path}")
    print(f"Saved: {hist_path}")
    print(f"Saved: {bins_csv}")
    print(f"Brier score: {brier:.4f}")

if __name__ == "__main__":
    main()
