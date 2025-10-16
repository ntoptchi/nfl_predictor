from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .config import Config

def split_train_val(df, holdout_season=None):
    holdout_season = holdout_season or Config.HOLDOUT_SEASON
    train_df = df[df["season"] < holdout_season].copy()
    val_df   = df[df["season"] == holdout_season].copy()
    return train_df, val_df

def build_xy(df):
    X = df[Config.FEATURES].copy()
    y = df[Config.TARGET].astype(int).copy()
    return X, y

def numeric_pipeline():
    # Fill NaNs (median), then scale
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

def build_preprocessor():
    numeric_features = Config.FEATURES
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline(), numeric_features)],
        remainder="drop",
    )
