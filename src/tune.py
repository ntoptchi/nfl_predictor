import os, json, optuna, numpy as np, pandas as pd
from argparse import ArgumentParser
from sklearn.base import clone
from sklearn.metrics import log_loss

from .config import Config
from .data import build_dataset
from .features import build_xy
from .model import build_ensemble


def _season_folds(df: pd.DataFrame, min_seasons=5, max_folds=None):
    seasons = sorted(df["season"].unique())
    if len(seasons) < min_seasons:
        raise RuntimeError(f"Not enough seasons ({len(seasons)}) for time-based CV.")
    tests = seasons[1:]
    if max_folds:
        tests = tests[-max_folds:]
    for test_season in tests:
        tr_idx = df.index[df["season"] < test_season].to_numpy()
        te_idx = df.index[df["season"] == test_season].to_numpy()
        if len(tr_idx) and len(te_idx):
            yield tr_idx, te_idx, test_season


def _suggest_params(trial: optuna.Trial) -> dict:
    # VotingClassifier nested params + weights
    params = {
        # LogisticRegression
        "clf__lr__C": trial.suggest_float("lr_C", 1e-3, 10.0, log=True),
        "clf__lr__penalty": "l2",

        # DecisionTree
        "clf__dt__max_depth": trial.suggest_int("dt_max_depth", 2, 20),
        "clf__dt__min_samples_split": trial.suggest_int("dt_min_split", 2, 20),
        "clf__dt__min_samples_leaf": trial.suggest_int("dt_min_leaf", 1, 10),

        # RandomForest
        "clf__rf__n_estimators": trial.suggest_int("rf_n_estimators", 150, 800, step=50),
        "clf__rf__max_depth": trial.suggest_int("rf_max_depth", 4, 24),
        "clf__rf__min_samples_split": trial.suggest_int("rf_min_split", 2, 30),
        "clf__rf__min_samples_leaf": trial.suggest_int("rf_min_leaf", 1, 10),
        "clf__rf__max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
    }

    # --- Estimator weights (LR, DT, RF) ---
    # Sample nonnegative weights; normalize to sum=1 (but sklearn doesn’t require it).
    w_lr = trial.suggest_float("w_lr", 0.0, 2.0)
    w_dt = trial.suggest_float("w_dt", 0.0, 2.0)
    w_rf = trial.suggest_float("w_rf", 0.0, 2.0)
    w = np.array([w_lr, w_dt, w_rf], dtype=float)
    if np.allclose(w.sum(), 0.0):
        w[:] = 1.0  # avoid all-zero; fallback to uniform
    w = (w / w.sum()).tolist()
    params["clf__weights"] = w

    return params


def _evaluate_logloss(model, X, y, folds):
    losses = []
    for tr_idx, te_idx, season in folds:
        m = clone(model)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        proba = m.predict_proba(X.iloc[te_idx])[:, 1]
        proba = np.clip(proba, 1e-6, 1 - 1e-6)
        losses.append(log_loss(y.iloc[te_idx], proba))
    return float(np.mean(losses))


def main():
    ap = ArgumentParser()
    ap.add_argument("--trials", type=int, default=getattr(Config, "TUNE_TRIALS", 40))
    ap.add_argument("--timeout", type=int, default=getattr(Config, "TUNE_TIMEOUT", None))
    ap.add_argument("--max-folds", type=int, default=None, help="Limit to latest N seasonal folds")
    ap.add_argument("--rolling-n", type=int, default=Config.ROLLING_N)
    ap.add_argument("--save", type=str, default=getattr(Config, "BEST_PARAMS_PATH", "artifacts/best_params.json"))
    args = ap.parse_args()

    # Build data across configured seasons
    df = build_dataset(seasons=Config.SEASONS, rolling_n=args.rolling_n)
    X, y = build_xy(df)
    folds = list(_season_folds(df, min_seasons=getattr(Config, "TUNE_MIN_SEASONS", 5), max_folds=args.max_folds))

    # Build a base ensemble WITHOUT loading any prior tuned params
    base_model = build_ensemble(params={})

    def objective(trial: optuna.Trial):
        params = _suggest_params(trial)
        m = clone(base_model)
        m.set_params(**params)
        return _evaluate_logloss(m, X, y, folds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    print("Best value (logloss):", study.best_value)
    print("Best params:", study.best_params)

    best = {
        "clf__lr__C": study.best_params["lr_C"],
        "clf__lr__penalty": "l2",
        "clf__dt__max_depth": study.best_params["dt_max_depth"],
        "clf__dt__min_samples_split": study.best_params["dt_min_split"],
        "clf__dt__min_samples_leaf": study.best_params["dt_min_leaf"],
        "clf__rf__n_estimators": study.best_params["rf_n_estimators"],
        "clf__rf__max_depth": study.best_params["rf_max_depth"],
        "clf__rf__min_samples_split": study.best_params["rf_min_split"],
        "clf__rf__min_samples_leaf": study.best_params["rf_min_leaf"],
        "clf__rf__max_features": study.best_params["rf_max_features"],
        "clf__weights": [
            study.best_params["w_lr"],
            study.best_params["w_dt"],
            study.best_params["w_rf"],
        ],
    }

    # Normalize weights before saving (nice to keep)
    w = np.array(best["clf__weights"], dtype=float)
    if not np.allclose(w.sum(), 0.0):
        best["clf__weights"] = (w / w.sum()).tolist()
    else:
        best["clf__weights"] = [1/3, 1/3, 1/3]

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    print(f"Saved best params -> {args.save}")


if __name__ == "__main__":
    main()
