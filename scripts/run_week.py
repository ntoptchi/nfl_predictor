import os, sys, subprocess
from argparse import ArgumentParser

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import Config

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--rolling-window", type=int, default=5, help="EPA/SR rolling window")
    ap.add_argument("--out", type=str, default="reports/picks.md")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 1) Market odds (requires ODDS_API_KEY; skip if not set)
    if os.getenv("ODDS_API_KEY"):
        run([sys.executable, "scripts/fetch_odds.py", "--season", str(args.season), "--week", str(args.week), "--out", os.path.join("data","market_odds.csv")])
    else:
        print("ODDS_API_KEY not set — skipping market odds fetch.")

    # 2) Weather
    run([sys.executable, "scripts/fetch_weather_nflweather.py", "--season", str(args.season), "--week", str(args.week), "--out", os.path.join("data","weather.csv")])

    # 3) QB change flags (derive across a range)
    run([sys.executable, "scripts/derive_qb_status.py", "--seasons", f"2015-{args.season}", "--out", os.path.join("data","qb_status.csv")])

    # 4) EPA/SR (derive across a range)
    run([sys.executable, "scripts/derive_epa_sr.py", "--seasons", f"2015-{args.season}", "--window", str(args.rolling_window), "--out", os.path.join("data","team_epa.csv")])

    # 5) Train (calibrated) & Predict
    run([sys.executable, "-m", "src.train"])
    run([sys.executable, "-m", "src.predict", "--season", str(args.season), "--week", str(args.week), "--out", args.out])

    print(f"All done. Picks -> {args.out}")

if __name__ == "__main__":
    main()
