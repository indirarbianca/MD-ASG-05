"""
training.py - Model training using sklearn Pipeline (outputs a single .pkl)
"""
 
import pickle
import optuna
from optuna.samplers import TPESampler
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
 
from source.config import RANDOM_STATE, N_TRIALS, N_CV_SPLITS, PIPELINE_PATH
from source.preprocessing import build_preprocessor
 
optuna.logging.set_verbosity(optuna.logging.WARNING)
 
 
def build_full_pipeline(model_params: dict) -> Pipeline:
    """Build full sklearn Pipeline: preprocessor + Logistic Regression."""
    preprocessor = build_preprocessor()
    model = LogisticRegression(**model_params, random_state=RANDOM_STATE)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
 
 
def objective_lr(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """Optuna objective function for Logistic Regression."""
    params = {
        "C": trial.suggest_float("C", 0.001, 100, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
        "max_iter": trial.suggest_int("max_iter", 100, 2000),
    }
 
    cv = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pipeline = build_full_pipeline(params)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return scores.mean()
 
 
def optimize_logistic_regression(X: pd.DataFrame, y: pd.Series, n_trials: int = N_TRIALS) -> dict:
    """Run Optuna hyperparameter search. Returns best params."""
    print(f"[training] Optimizing Logistic Regression with {n_trials} trials...")
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: objective_lr(trial, X, y), n_trials=n_trials, show_progress_bar=True)
    print(f"[training] Best CV Accuracy: {study.best_value:.4f}")
    print(f"[training] Best params: {study.best_params}")
    return study.best_params
 
 
def train_pipeline(X: pd.DataFrame, y: pd.Series, params: dict) -> Pipeline:
    """Train full sklearn Pipeline on the entire dataset."""
    pipeline = build_full_pipeline(params)
    pipeline.fit(X, y)
    print("[training] Pipeline trained on full dataset.")
    return pipeline
 
 
def save_pipeline(pipeline: Pipeline, filepath=PIPELINE_PATH) -> None:
    """Save the full pipeline to a single pickle file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"[training] Pipeline saved to {filepath}")
 
 
def load_pipeline(filepath=PIPELINE_PATH) -> Pipeline:
    """Load the full pipeline from a pickle file."""
    with open(filepath, "rb") as f:
        pipeline = pickle.load(f)
    print(f"[training] Pipeline loaded from {filepath}")
    return pipeline