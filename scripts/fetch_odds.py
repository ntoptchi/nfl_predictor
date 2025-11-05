import os, sys, requests, pandas as pd
from argparse import ArgumentParser

# allow importing project modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import load_games
from src.config import Config

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
SPORT = "americanfootball_nfl"
BASE = "https://api.the-odds-api.com/v4/sports"

# Map TheOddsAPI names to nfl_data_py abbreviations
NAME_TO_ABBR = {
    "New England Patriots":"NE","New York Jets":"NYJ","Kansas City Chiefs":"KC","Baltimore Ravens":"BAL",
    "Buffalo Bills":"BUF","Miami Dolphins":"MIA","Cleveland Browns":"CLE","Cincinnati Bengals":"CIN",
    "Pittsburgh Steelers":"PIT","Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX",
    "Tennessee Titans":"TEN","Denver Broncos":"DEN","Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC",
    "Dallas Cowboys":"DAL","Philadelphia Eagles":"PHI","New York Giants":"NYG","Washington Commanders":"WAS",
    "Green Bay Packers":"GB","Chicago Bears":"CHI","Minnesota Vikings":"MIN","Detroit Lions":"DET",
    "Atlanta Falcons":"ATL","Carolina Panthers":"CAR","New Orleans Saints":"NO","Tampa Bay Buccaneers":"TB",
    "San Francisco 49ers":"SF","Seattle Seahawks":"SEA","Los Angeles Rams":"LA","Arizona Cardinals":"ARI",
}

def _fetch_current_odds(region="us", markets=("spreads","h2h","totals")):
    if not ODDS_API_KEY:
        raise SystemExit("Set ODDS_API_KEY env var first (or disable USE_MARKET in config).")
    params = {"regions": region, "markets": ",".join(markets),
              "oddsFormat": "american", "dateFormat": "unix", "apiKey": ODDS_API_KEY}
    url = f"{BASE}/{SPORT}/odds"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def _pick_book(bookmakers):
    priority = ["draftkings","fanduel","betmgm","caesars"]
    bykey = {b["key"]: b for b in bookmakers}
    for k in priority:
        if k in bykey: return bykey[k]
    return bookmakers[0] if bookmakers else None

def _parse_event(ev):
    home_name, away_name = ev.get("home_team"), ev.get("away_team")
    if home_name not in NAME_TO_ABBR or away_name not in NAME_TO_ABBR:
        return None
    home, away = NAME_TO_ABBR[home_name], NAME_TO_ABBR[away_name]

    bm = _pick_book(ev.get("bookmakers", []))
    if not bm: return None
    mkts = {m["key"]: m for m in bm.get("markets", [])}

    spread_home = ml_home = ml_away = total = None

    if "spreads" in mkts:
        for out in mkts["spreads"]["outcomes"]:
            if out["name"] == home_name:
                spread_home = out.get("point")
                break

    if "h2h" in mkts:
        for out in mkts["h2h"]["outcomes"]:
            if out["name"] == home_name: ml_home = out.get("price")
            if out["name"] == away_name: ml_away = out.get("price")

    if "totals" in mkts and mkts["totals"]["outcomes"]:
        total = mkts["totals"]["outcomes"][0].get("point")

    return {"home_team": home, "away_team": away,
            "close_spread_home": spread_home, "ml_home": ml_home, "ml_away": ml_away,
            "total_points": total}

def main():
    ap = ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(Config.DATA_DIR, Config.FILE_MARKET))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    odds = _fetch_current_odds()
    rows = []
    for ev in odds:
        rec = _parse_event(ev)
        if rec: rows.append(rec)
    if not rows:
        print("No odds parsed — check API key / quota.")
        return

    df = pd.DataFrame(rows).drop_duplicates(subset=["home_team","away_team"], keep="last")

    # attach game_id via schedule for this season+week
    sched = load_games([args.season])
    wk = sched[(sched["season"]==args.season) & (sched["week"]==args.week)]
    merged = df.merge(wk[["game_id","home_team","away_team"]], on=["home_team","away_team"], how="inner")
    if merged.empty:
        print("No games matched to schedule — names may differ.")
        return

    out_cols = ["game_id","close_spread_home","ml_home","ml_away","total_points"]
    merged = merged[out_cols].drop_duplicates(subset=["game_id"], keep="last")

    # append/overwrites per game_id
    if os.path.exists(args.out):
        old = pd.read_csv(args.out)
        merged = pd.concat([old, merged]).drop_duplicates(subset=["game_id"], keep="last")
    merged.to_csv(args.out, index=False)
    print(f"Saved {len(merged)} rows -> {args.out}")

if __name__ == "__main__":
    main()
