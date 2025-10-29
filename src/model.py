import os, json
import numpy as np
from joblib import dump, load

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from .config import Config
from .features import build_preprocessor

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


def build_base_estimators():
    models = []
    models.append(("lr", LogisticRegression(max_iter=1000, random_state=Config.RANDOM_STATE)))
    models.append(("dt", DecisionTreeClassifier(max_depth=6, random_state=Config.RANDOM_STATE)))
    models.append(("rf", RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        random_state=Config.RANDOM_STATE, n_jobs=-1)))
    if getattr(Config, "USE_XGBOOST", False) and XGBClassifier is not None:
        models.append(("xgb", XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=Config.RANDOM_STATE, n_jobs=-1, tree_method="hist"
        )))
    return models


def _load_best_params():
    path = getattr(Config, "BEST_PARAMS_PATH", None)
    if not path:
        return None
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _align_weights(weights, n_estimators):
    """Ensure weights length matches number of estimators; pad/truncate; avoid all-zero."""
    if weights is None:
        return None
    w = list(weights)
    if len(w) < n_estimators:
        w = w + [1.0] * (n_estimators - len(w))
    elif len(w) > n_estimators:
        w = w[:n_estimators]
    # avoid all-zero
    if sum(abs(x) for x in w) == 0:
        w = [1.0] * n_estimators
    return w

def build_ensemble(preprocessor=None, params: dict | None = None):
    pre = preprocessor or build_preprocessor()
    estimators = build_base_estimators()

    # Apply tuned params (from arg or artifacts/best_params.json)
    tuned = params or _load_best_params() or {}

    # If tuned has weights, align them to current estimator count
    if "clf__weights" in tuned:
        tuned = dict(tuned)  # shallow copy
        tuned["clf__weights"] = _align_weights(tuned["clf__weights"], len(estimators))

    voter = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1,
                             weights=tuned.get("clf__weights", None))
    pipe = Pipeline([("pre", pre), ("clf", voter)])

    # Remove weights key before set_params (we already passed it to VotingClassifier init)
    tuned.pop("clf__weights", None)
    if tuned:
        pipe.set_params(**tuned)
    return pipe


def calibrate_model(trained_pipe, X_val, y_val, method="isotonic"):
    """Wrap a prefit pipeline with a probability calibrator, fit on holdout only."""
    return CalibratedClassifierCV(trained_pipe, method=method, cv="prefit").fit(X_val, y_val)


def save_model(pipe):
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    dump(pipe, Config.MODEL_PATH)


def load_model():
    return load(Config.MODEL_PATH)
