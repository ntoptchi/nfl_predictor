import pandas as pd
from sklearn.metrics import accuracy_score
from .data import build_dataset
from .features import build_xy
from .model import build_ensemble
from .config import Config

def rolling_backtest(start=2015, end=2024):
    results = []
    for test_season in range(max(start+2, start), end+1):
        seasons = list(range(start, test_season+1))
        df = build_dataset(seasons=seasons, rolling_n=Config.ROLLING_N)
        train_df = df[df["season"] < test_season]
        test_df  = df[df["season"] == test_season]
        X_tr, y_tr = build_xy(train_df)
        X_te, y_te = build_xy(test_df)
        m = build_ensemble(); m.fit(X_tr, y_tr)
        acc = (m.predict(X_te) == y_te).mean()
        results.append((test_season, acc))
    return results

def main():
    res = rolling_backtest(start=2015, end=Config.HOLDOUT_SEASON)
    for yr, acc in res:
        print(f"{yr}: acc={acc:.3f}")
    mean_acc = sum(a for _, a in res)/len(res)
    print(f"Mean rolling acc: {mean_acc:.3f}")


if __name__ == "__main__":
    main()