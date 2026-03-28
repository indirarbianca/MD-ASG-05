"""
preprocessing.py - Feature engineering and preprocessing for Spaceship Titanic
Uses OrdinalEncoder + ColumnTransformer from sklearn.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from source.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    SPENDING_COLS,
    SPENDING_RATIO_COLS,
)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering and return enriched DataFrame."""
    df = df.copy()

    # extract features from cabin
    df["Deck"] = df["Cabin"].apply(lambda x: x.split("/")[0] if pd.notna(x) else np.nan)
    df["Cabin_num"] = df["Cabin"].apply(lambda x: x.split("/")[1] if pd.notna(x) else np.nan).astype(float)
    df["Side"] = df["Cabin"].apply(lambda x: x.split("/")[2] if pd.notna(x) else np.nan)

    # Group info from PassengerId
    df["Group"] = df["PassengerId"].apply(lambda x: x.split("_")[0])
    df["Group_size"] = df.groupby("Group")["Group"].transform("count")
    df["Solo"] = (df["Group_size"] == 1).astype(int)

    # name features
    df["LastName"] = df["Name"].apply(lambda x: x.split()[-1] if pd.notna(x) else np.nan)
    df["Family_size"] = df.groupby("LastName")["LastName"].transform("count")

    # spending features
    df["TotalSpending"] = df[SPENDING_COLS].sum(axis=1)
    df["HasSpending"] = (df["TotalSpending"] > 0).astype(int)
    df["NoSpending"] = (df["TotalSpending"] == 0).astype(int)
    for col in SPENDING_COLS:
        df[f"{col}_ratio"] = df[col] / (df["TotalSpending"] + 1)

    # age groups
    df["Age_group"] = pd.cut(
        df["Age"], bins=[0, 12, 18, 30, 50, 100],
        labels=["Child", "Teen", "Young_Adult", "Adult", "Senior"],
    ).astype(str)

    # missing indicators (before converting bool to str)
    df["Age_missing"] = df["Age"].isna().astype(int)
    df["CryoSleep_missing"] = df["CryoSleep"].isna().astype(int)

    # convert bool columns to string so OrdinalEncoder doesn't get mixed types
    for col in ["CryoSleep", "VIP"]:
        df[col] = df[col].astype(str)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Build and return a sklearn ColumnTransformer for preprocessing."""
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ("num", numerical_pipeline, NUMERICAL_FEATURES + SPENDING_RATIO_COLS),
    ])

    return preprocessor


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target y from training DataFrame."""
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + SPENDING_RATIO_COLS
    X = df[feature_cols]
    y = df["Transported"].astype(int)
    return X, y


def get_X(df: pd.DataFrame) -> pd.DataFrame:
    """Extract feature matrix X from test DataFrame."""
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + SPENDING_RATIO_COLS
    return df[feature_cols]