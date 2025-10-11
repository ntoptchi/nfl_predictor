import pandas as pd
import numpy as np
from dateutil import parser
from .config import Config
import nfl_data_py as nfl

def load_games(seasons):
    #Schedule with results
    sched = nfl.import_schedules(seasons)
    #Standardize key fields
    sched =sched.rename(columns={
        "home_team": "home_team",
        "away_team": "away_team",
        "home_score": "home_score",
        "away_score": "away_score",
        "gameday": "gameday",
        "game_id": "game_id",
        "season": "season",
        "week": "week",
        "game_type": "game_type",
        "venue": "venue"
    })
    #Parse dates of games
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    #Filter regular season only
    sched = sched[sched["game_type"] == "REG"].copy()
    #Winner flag
    sched["home_win"] = (sched["home_score"] > sched["away_score"]).astype(int)
    sched["away_win"] = 1 - sched["home_win"]
    return sched