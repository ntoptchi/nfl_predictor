import os, sys, json, re, requests
import pandas as pd
from argparse import ArgumentParser

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import load_games
from src.config import Config

NFLWEATHER_WEEK_URL = "https://www.nflweather.com/api1/week"

PRECIP_KEYWORDS = re.compile(r"rain|snow|drizzle|thunder|storm|sleet|showers", re.I)

def _to_int(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        s = str(x).strip()
        # strip units if present (e.g., "12 mph", "61°F")
        s = re.sub(r"[^0-9\-\.\+]", "", s)
        if s == "" or s == "." or s == "-":
            return None
        return int(round(float(s)))
    except Exception:
        return None

def _is_precip(text):
    if text is None:
        return 0
    return 1 if PRECIP_KEYWORDS.search(str(text)) else 0

def _is_outdoor(stadium_type):
    if stadium_type is None:
        return 1  # default to outdoor if unknown
    s = str(stadium_type).lower()
    # nflweather often uses: "Outdoor", "Dome", "Dome (Open)", "Retractable Roof"
    if "outdoor" in s or "open" in s:
        return 1
    return 0

def fetch_week(season: int, week: int) -> pd.DataFrame:
    """Fetch one week's weather from nflweather.com. Returns rows keyed by (home_team, away_team)."""
    params = {"season": season, "week": week}
    r = requests.get(NFLWEATHER_WEEK_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected nflweather payload: {type(data)}")

    rows = []
    for g in data:
        # nflweather fields vary slightly; be flexible
        home = g.get("home_team") or g.get("home") or g.get("home_abbr")
        away = g.get("away_team") or g.get("away") or g.get("away_abbr")
        temp = g.get("temperature") or g.get("temp")
        wind = g.get("wind") or g.get("wind_mph")
        wx   = g.get("weather") or g.get("conditions")
        stad = g.get("stadium_type") or g.get("stadiumType")

        if not home or not away:
            # skip if we can’t identify teams
            continue

        rows.append({
            "home_team": str(home).strip(),
            "away_team": str(away).strip(),
            "temp_f": _to_int(temp),
            "wind_mph": _to_int(wind),
            "is_precip": _is_precip(wx),
            "is_outdoor": _is_outdoor(stad),
        })

    return pd.DataFrame(rows)

def attach_game_ids(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Join to your schedule to get game_id."""
    sched = load_games([season])
    wk = sched[(sched["season"] == season) & (sched["week"] == week)].copy()

    # Try exact join first
    merged = df.merge(
        wk[["game_id", "home_team", "away_team"]],
        on=["home_team", "away_team"],
        how="left",
        validate="many_to_one"
    )

    # If some game_id are missing, try swapped (bad home/away in feed)
    missing = merged["game_id"].isna()
    if missing.any():
        swapped = df.loc[missing, :].rename(columns={"home_team": "away_team", "away_team": "home_team"})
        remerge = swapped.merge(
            wk[["game_id", "home_team", "away_team"]],
            on=["home_team", "away_team"],
            how="left",
            validate="many_to_one"
        )
        merged.loc[missing, "game_id"] = remerge["game_id"].values

    # Keep only rows with a matched game_id
    merged = merged.dropna(subset=["game_id"]).copy()
    merged["game_id"] = merged["game_id"].astype(str)
    return merged

def main():
    ap = ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(Config.DATA_DIR, Config.FILE_WEATHER))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    week_df = fetch_week(args.season, args.week)
    if week_df.empty:
        print("No weather rows returned for that week.")
        return

    week_df = attach_game_ids(week_df, args.season, args.week)
    if week_df.empty:
        print("Could not match any games to game_id (team codes mismatch?).")
        return

    out_cols = ["game_id", "wind_mph", "temp_f", "is_precip", "is_outdoor"]
    week_df = week_df[out_cols].drop_duplicates(subset=["game_id"], keep="last")

    # merge/append with existing file
    if os.path.exists(args.out):
        old = pd.read_csv(args.out)
        both = pd.concat([old, week_df], ignore_index=True)
        both = both.drop_duplicates(subset=["game_id"], keep="last")
        both.to_csv(args.out, index=False)
        print(f"Updated {args.out} (now {len(both)} rows)")
    else:
        week_df.to_csv(args.out, index=False)
        print(f"Saved {len(week_df)} rows -> {args.out}")

if __name__ == "__main__":
    main()
