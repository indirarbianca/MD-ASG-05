"""
ingest.py - Data ingestion for Spaceship Titanic
"""

import pandas as pd
from source.config import TRAIN_PATH, TEST_PATH


def load_train_data(filepath=TRAIN_PATH) -> pd.DataFrame:
    """Load training data from CSV file."""
    df = pd.read_csv(filepath)
    print(f"[ingest] Training data loaded: {df.shape}")
    return df


def load_test_data(filepath=TEST_PATH) -> pd.DataFrame:
    """Load test data from CSV file."""
    df = pd.read_csv(filepath)
    print(f"[ingest] Test data loaded: {df.shape}")
    return df


def get_data_summary(df: pd.DataFrame) -> None:
    """Print basic summary of the dataset."""
    print(f"\nShape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    if "Transported" in df.columns:
        print(f"\nTarget distribution:\n{df['Transported'].value_counts()}")
