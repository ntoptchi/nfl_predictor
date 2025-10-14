from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from joblib import dump, load
from .config import Config
from .features import build_preprocessor
import numpy as np

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

def build_base_estimators():
    models = []
    models.append(("lr", LogisticRegression(max_iter=1000, random_state=Config.RANDOM_STATE)))
    models.append(("dt", DecisionTreeClassifier(max_depth=6, random_state=Config.RANDOM_STATE)))
    models.append(("rf", RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2, random_state=Config.RANDOM_STATE)))
    
    if Config.USE_XGBOOST and XGBClassifier is not None:
        models.append(("xgb", XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, random_state=Config.RANDOM_STATE, n_jobs=-1, tree_method="hist"
        )))
        return models

def build_ensemble(preprocessor=None):
    pre = preprocessor or build_preprocessor()
    estimators = build_base_estimators()
    # soft voting for calibrated probs
    voter = VotingClassifier(estimators=estimators, voting="soft")
    pipe = Pipeline([("pre", pre), ("clf", voter)])
    return pipe

def save_model(pipe):
    dump(pipe, Config.MODEL_PATH)

def load_model():
    return load(Config.MODEL_PATH)