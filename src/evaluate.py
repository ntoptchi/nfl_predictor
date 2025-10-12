import pandas as pd
from sklearn.metrics import accuracy_score
from .data import build_dataset
from .features import build_xy
from .model import build_ensemble
from .config import Config

def backtest(last_k_seasons=3):
    seasons = sorted(Config.SEASONS)[-last_k_seasons-1:] #includes history for rolls
    df = build_dataset(seasons=seasons,rolling_n=Config.ROLLING_N)
    #train on first (k-1) seasons, test on last season
    last_season = max(df["season"])
    train_df = df[df["season"] < last_season]
    test_df = df[df["season"] == last_season]
    X_tr, y_tr = build_xy(train_df)
    X_te, y_te = build_xy(test_df)
    
    model = build_ensemble()
    model.fit(X_tr, y_tr)
    yhat = model.predict(X_te)
    acc = accuracy_score(y_te, yhat)
    return last_season, acc

def main():
    last_season, acc = backtest(last_k_seasons=4)
    print(f"Backtest on season {last_season}: accuracy={ac:.3f}")

if __name__ == "__main__":
    main()