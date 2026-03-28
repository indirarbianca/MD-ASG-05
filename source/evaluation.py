"""
evaluation.py - Model evaluation utilities for Spaceship Titanic
"""
 
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
 
from source.config import RANDOM_STATE, N_CV_SPLITS
 
 
def evaluate_on_validation(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str = "Model",
) -> dict:
    """Evaluate pipeline on a held-out validation set and print a full report."""
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else None
 
    acc = accuracy_score(y_val, y_pred)
    roc = roc_auc_score(y_val, y_proba) if y_proba is not None else None
 
    print(f"\n{'='*40}")
    print(f"  {model_name} Evaluation")
    print(f"{'='*40}")
    print(f"  Accuracy : {acc:.4f}")
    if roc:
        print(f"  ROC-AUC  : {roc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_val, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")
 
    return {"accuracy": acc, "roc_auc": roc}
 
 
def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_CV_SPLITS,
    model_name: str = "Model",
) -> dict:
    """Run stratified k-fold cross-validation and print a summary."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
 
    print(f"\n[evaluation] {model_name} CV Results ({n_splits}-fold):")
    print(f"  Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    return {"cv_mean": scores.mean(), "cv_std": scores.std(), "cv_scores": scores}