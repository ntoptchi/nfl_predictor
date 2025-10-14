import argparse
import pandas as pd
from .data import build_dataset, load_games
from .model import load_model
from .config import Config

def week_matchups(season, week):
    sched = load_games([season])
    wk = sched[(sched["season"] == season) & (sched["week"] == week)].copy()
    return wk[["game_id", "home_team", "away_team", "gameday", "week", "season"]]