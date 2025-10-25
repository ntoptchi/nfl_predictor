import os, sys, pandas as pd, numpy as np
from argparse import ArgumentParser

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import nfl_data_py as nfl

def _team_week_epa_sr(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    # keep snaps with EPA defined
    df = df[df["epa"].notna()].copy()

    # success flag: nflfastR defines 'success' column; fall back to epa>0 if missing
    if "success" not in df.columns:
        df["success"] = (df["epa"] > 0).astype(int)

    # OFFENSE aggregates
    off = (df.groupby(["season","week","posteam"], as_index=False)
             .agg(off_epa_mean=("epa","mean"),
                  sr_off=("success","mean"))
             .rename(columns={"posteam":"team"}))

    # DEFENSE aggregates — use defending team perspective
    dff = (df.groupby(["season","week","defteam"], as_index=False)
             .agg(def_epa_mean=("epa","mean"),
                  sr_def=("success","mean"))
             .rename(columns={"defteam":"team"}))

    out = off.merge(dff, on=["season","week","team"], how="outer")
    # ensure numeric
    for c in ["off_epa_mean","def_epa_mean","sr_off","sr_def"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

def _rolling_shifted(df, window):
    df = df.sort_values(["team","season","week"]).copy()
    g = df.groupby("team", group_keys=False)
    for raw, roll in [
        ("off_epa_mean","off_epa_roll"),
        ("def_epa_mean","def_epa_roll"),
        ("sr_off","sr_off_roll"),
        ("sr_def","sr_def_roll"),
    ]:
        df[roll] = g[raw].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    return df

def main():
    ap = ArgumentParser()
    ap.add_argument("--seasons", type=str, default="2015-2024", help="e.g. 2018-2024")
    ap.add_argument("--window", type=int, default=5, help="rolling window (games)")
    ap.add_argument("--out", default=os.path.join("data","team_epa.csv"))
    args = ap.parse_args()

    start, end = [int(x) for x in args.seasons.split("-")]
    seasons = list(range(start, end+1))

    print(f"Downloading pbp for {seasons} ...")
    pbp = nfl.import_pbp_data(seasons)

    print("Aggregating EPA/SR by team-week ...")
    teamwk = _team_week_epa_sr(pbp)

    print(f"Rolling (window={args.window}) with shift=1 ...")
    teamwk = _rolling_shifted(teamwk, args.window)

    out_cols = ["season","week","team","off_epa_roll","def_epa_roll","sr_off_roll","sr_def_roll"]
    teamwk = teamwk[out_cols].copy()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    teamwk.to_csv(args.out, index=False)
    print(f"Saved {len(teamwk)} rows -> {args.out}")

if __name__ == "__main__":
    main()
