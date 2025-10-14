import argparse
import pandas as pd
from .data import build_dataset, load_games
from .model import load_model
from .config import Config

def week_matchups(season, week):
    sched = load_games([season])
    wk = sched[(sched["season"] == season) & (sched["week"] == week)].copy()
    return wk[["game_id", "home_team", "away_team", "gameday", "week", "season"]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=Config.CURRENT_SEASON)
    parser.add_argument("--week", type=int, required=True)
    args = parser,parse_args()

    # Build dataset for all seasons including current so rolling features exist
    df = build_dataset(seasons=list(range(2015, args.season + 1)), rolling_n=Config.ROLLING_N)

    # Filter to target week games (both team perspectives produce two rows per game)
    games = week_matchups(args.season, args.week)
    df_wk = df.merge(games[["game_id"]], on="game_id", how="inner").copy()

    model = load_model()
    X = df_wk[Config.FEATURES]
    # prob team_win=1
    proba = model.predict_proba(X)[:,1]
    df_wk["team_prob_win"] = proba

    # Aggregate to a single prediction per game_id: choose the (team,opp) with higher prob
    agg = (df_wk.sort_values(["game_id","team_prob_win"], ascending=[True, False])
               .groupby("game_id").head(1))

  # Map back to human-readable pick (winner)
    # If 'home'==1 in that row, winner is home_team; 
    # else away_team (since we stacked both perspectives)
    picks = agg.merge(games, on="game_id", how="left")
    picks["predicted_winner"] = picks.apply(
        lambda r: r["home_team"] if r["home"] == 1 else r["away_team"], axis=1
    )

    out = picks[["season","week","gameday","home_team","away_team","predicted_winner","team_prob_win"]]
    out = out.sort_values("gameday")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()