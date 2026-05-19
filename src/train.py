import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import StackingRegressor, StackingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# Import our preprocessing logic so training has fresh data
from preprocessing import prepare_and_save_data

# Load MLflow Tracking Environment
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")

def build_regression_ensemble():
    """Combines XGBoost and LightGBM for continuous price prediction."""
    base_estimators = [
        ("xgb", XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)),
        ("lgbm", LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)),
        ("rf", RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42))
    ]
    # Meta-learner to intelligently weigh the base predictions
    meta_model = RidgeCV() 
    return StackingRegressor(estimators=base_estimators, final_estimator=meta_model, cv=3)

def build_classification_ensemble():
    """Combines XGBoost and LightGBM for categorical price tier prediction."""
    base_estimators = [
        ("xgb", XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric="mlogloss")),
        ("lgbm", LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42))
    ]
    # Meta-learner for classification
    meta_model = LogisticRegression()
    return StackingClassifier(estimators=base_estimators, final_estimator=meta_model, cv=3)

def train_and_log():
    """Executes the training sequence and logs everything to MLflow."""
    print("Executing Preprocessing pipeline...")
    data = prepare_and_save_data()
    
    if data is None:
        print("Pipeline aborted due to insufficient data.")
        return
        
    X_train, X_test = data["X_train"], data["X_test"]
    y_train_reg, y_test_reg = data["y_train_reg"], data["y_test_reg"]
    y_train_clf, y_test_clf = data["y_train_clf"], data["y_test_clf"]
    
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Real_Estate_Intelligence_Ensemble")
    
    with mlflow.start_run(run_name="Automated_Retraining_Run"):
        print("Training Regression Ensemble...")
        reg_model = build_regression_ensemble()
        reg_model.fit(X_train, y_train_reg)
        
        print("Training Classification Ensemble...")
        clf_model = build_classification_ensemble()
        clf_model.fit(X_train, y_train_clf)
        
        # Evaluation 
        print("Evaluating Models...")
        reg_preds = reg_model.predict(X_test)
        clf_preds = clf_model.predict(X_test)
        
        # Calculate Regression Metrics
        rmse = mean_squared_error(y_test_reg, reg_preds, squared=False)
        mae = mean_absolute_error(y_test_reg, reg_preds)
        r2 = r2_score(y_test_reg, reg_preds)
        
        # Calculate Classification Metrics
        acc = accuracy_score(y_test_clf, clf_preds)
        f1 = f1_score(y_test_clf, clf_preds, average="weighted")
        
        # --- MLflow Logging ---
        print("Logging metrics and models to MLflow...")
        # 1. Log Metrics
        mlflow.log_metrics({
            "reg_rmse": rmse,
            "reg_mae": mae,
            "reg_r2": r2,
            "clf_accuracy": acc,
            "clf_f1": f1
        })
        
        # 2. Log Model Artifacts (The actual .pkl files are securely stored by MLflow)
        mlflow.sklearn.log_model(reg_model, "regression_ensemble")
        mlflow.sklearn.log_model(clf_model, "classification_ensemble")
        
        # 3. Save locally for the FastAPI Container to pick up during deployment
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(reg_model, "artifacts/regression_model.joblib")
        joblib.dump(clf_model, "artifacts/classification_model.joblib")
        
        print(f"Training Complete! Metrics -> R2: {r2:.3f} | Accuracy: {acc:.3f}")

if __name__ == "__main__":
    train_and_log()
