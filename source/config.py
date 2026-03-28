"""
config.py - Central configuration for Spaceship Titanic pipeline
"""

from pathlib import Path

# paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
PIPELINE_PATH = MODELS_DIR / "pipeline.pkl"
SUBMISSION_PATH = ROOT_DIR / "submission.csv"

# reproducibility
RANDOM_STATE: int = 42

# features 
CATEGORICAL_FEATURES: list[str] = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "Age_group"]
NUMERICAL_FEATURES: list[str] = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "Cabin_num", "Group_size", "Solo", "Family_size", "TotalSpending",
    "HasSpending", "NoSpending", "Age_missing", "CryoSleep_missing",
]
SPENDING_COLS: list[str] = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
SPENDING_RATIO_COLS: list[str] = [f"{col}_ratio" for col in SPENDING_COLS]

# training 
N_TRIALS: int = 30
N_CV_SPLITS: int = 5
TEST_SIZE: float = 0.2
