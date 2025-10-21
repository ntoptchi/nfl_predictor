import argparse
import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score
from .data import build_dataset
from .features import split_train_val, build_xy
from .model import build_ensemble, save_model
from .config import Config
from .model import calibrate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling_n", type=int, default = Config.ROLLING_N)
    args = parser.parse_args()

    df = build_dataset(seasons=Config.SEASONS, rolling_n = args.rolling_n)
    train_df,val_df = split_train_val(df, holdout_season = Config.HOLDOUT_SEASON)

    X_train, y_train = build_xy(train_df)
    X_val, y_val = build_xy(val_df)

    model = build_ensemble()
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)


    cal_model = calibrate_model(model, X_val, y_val, method="isotonic")
    save_model(cal_model)

    acc = accuracy_score(y_val,val_pred)
    print(f"Holdout season {Config.HOLDOUT_SEASON} accuracy: {acc:.3f}")

    if getattr(Config, "CALIBRATE", False):
        method = getattr(Config, "CALIBRATION_METHOD", "isotonic")
        model = calibrate_model(model, X_val, y_val, method=method)
        print(f"Calibrated probabilities using {method}.")

    save_model(model)
    print(f"Saved model to {Config.MODEL_PATH}")

if __name__ == "__main__":
    main()