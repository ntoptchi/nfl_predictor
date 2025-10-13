import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .config import Config

def split_train_val(df, holdout_season=None):
    holdout_season = holdout_season or Config.HOLDOUT_SEASON
    train.df = df[df["season"] < holdout_season].copy()
    val.df = df[df["season"] == holdout_season].copy()
    return train_df, val_df

def build_xy(df):
    X = df[Config.FEATURES].copy()
    y =df[Config.TARGET].astype(int).copy()
    return X, y

def numeric_pipeline():
    return Pipeline([("scaler", StandardScaler())])