"""
pipeline.py - End-to-end ML pipeline orchestration for Spaceship Titanic
"""
 
import pandas as pd
from sklearn.model_selection import train_test_split
 
from source.config import (
    RANDOM_STATE, TEST_SIZE,
    TRAIN_PATH, TEST_PATH,
    PIPELINE_PATH, SUBMISSION_PATH,
)
from source.ingest import load_train_data, load_test_data
from source.preprocessing import feature_engineering, get_X_y, get_X
from source.training import optimize_logistic_regression, train_pipeline, save_pipeline, load_pipeline
from source.evaluation import evaluate_on_validation, cross_validate_pipeline
 
 
def run_training_pipeline(
    train_path=TRAIN_PATH,
    n_trials: int = 30,
) -> None:
    """Full training pipeline: ingest → feature engineering → train → evaluate → save."""
    print("\n" + "=" * 50)
    print("  SPACESHIP TITANIC - TRAINING PIPELINE")
    print("=" * 50)
 
    print("\n[1/4] Loading data...")
    train_df = load_train_data(train_path)
 
    print("\n[2/4] Feature engineering...")
    train_df = feature_engineering(train_df)
    X, y = get_X_y(train_df)
 
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
 
    print("\n[3/4] Optimizing and training...")
    best_params = optimize_logistic_regression(X, y, n_trials=n_trials)
    pipeline = train_pipeline(X, y, best_params)
    save_pipeline(pipeline, PIPELINE_PATH)
 
    print("\n[4/4] Evaluating...")
    evaluate_on_validation(pipeline, X_val, y_val, model_name="Logistic Regression (Optimized)")
    cross_validate_pipeline(pipeline, X, y, model_name="Logistic Regression (Optimized)")
 
    print("\nTraining pipeline complete!")
 
 
def run_prediction_pipeline(
    test_path=TEST_PATH,
    output_path=SUBMISSION_PATH,
) -> pd.DataFrame:
    """Prediction pipeline: ingest test → feature engineering → predict → save."""
    print("\n" + "=" * 50)
    print("  SPACESHIP TITANIC - PREDICTION PIPELINE")
    print("=" * 50)
 
    pipeline = load_pipeline(PIPELINE_PATH)
 
    test_df = load_test_data(test_path)
    test_passenger_ids = test_df["PassengerId"].copy()
    test_df = feature_engineering(test_df)
    X_test = get_X(test_df)
 
    predictions = pipeline.predict(X_test)
 
    submission = pd.DataFrame({
        "PassengerId": test_passenger_ids,
        "Transported": predictions.astype(bool),
    })
    submission.to_csv(output_path, index=False)
    print(f"[pipeline] Submission saved to {output_path} ({len(submission)} rows)")
    return submission
 
 
def predict_single(input_features: dict) -> bool:
    """Predict transport status for a single passenger. Used by the Streamlit app."""
    pipeline = load_pipeline(PIPELINE_PATH)
 
    row = {
        "PassengerId": "0000_01",
        "Name": "Unknown Unknown",
        "HomePlanet": input_features.get("HomePlanet", "Earth"),
        "CryoSleep": input_features.get("CryoSleep", False),
        "Cabin": input_features.get("Cabin", "F/100/S"),
        "Destination": input_features.get("Destination", "TRAPPIST-1e"),
        "Age": input_features.get("Age", 27.0),
        "VIP": input_features.get("VIP", False),
        "RoomService": input_features.get("RoomService", 0.0),
        "FoodCourt": input_features.get("FoodCourt", 0.0),
        "ShoppingMall": input_features.get("ShoppingMall", 0.0),
        "Spa": input_features.get("Spa", 0.0),
        "VRDeck": input_features.get("VRDeck", 0.0),
    }
 
    df = pd.DataFrame([row])
    df = feature_engineering(df)
    X = get_X(df)
    return bool(pipeline.predict(X)[0])
 
 
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description="Spaceship Titanic Pipeline")
    parser.add_argument("--mode", choices=["train", "predict", "all"], default="all")
    parser.add_argument("--n_trials", type=int, default=30)
    args = parser.parse_args()
 
    if args.mode in ("train", "all"):
        run_training_pipeline(n_trials=args.n_trials)
    if args.mode in ("predict", "all"):
        run_prediction_pipeline()
