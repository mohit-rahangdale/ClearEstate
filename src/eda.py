import os
import json
import pandas as pd
import mlflow
from sqlalchemy import create_engine
import boto3
# pyrefly: ignore [missing-import]
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TextOverviewPreset
from evidently.pipeline.column_mapping import ColumnMapping

# 1. Load Environment Variables
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
AWS_BUCKET = os.environ.get("AWS_S3_BUCKET")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")

DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URL)

def fetch_data_splits():
    """Fetches baseline and target data."""
    # In production, baseline is usually your current model's training dataset.
    # Here, we use rolling time windows.
    query_baseline = "SELECT * FROM properties WHERE fetched_at < NOW() - INTERVAL '7 days'"
    query_target = "SELECT * FROM properties WHERE fetched_at >= NOW() - INTERVAL '7 days'"
    return pd.read_sql(query_baseline, engine), pd.read_sql(query_target, engine)

def run_advanced_profiling(df_baseline, df_target):
    """Executes EDA, logs to MLflow, and determines if retraining is needed."""
    
    # 2. Tell the profiler what types of data we have (Crucial for NLP & Math)
    column_mapping = ColumnMapping(
        target="price", # Our regression target
        prediction=None,
        numerical_features=["bedrooms", "bathrooms", "sqft"],
        categorical_features=["neighborhood"],
        text_features=["description", "title"] # NLP specific profiling
    )

    # 3. Build the comprehensive report
    report = Report(metrics=[
        DataQualityPreset(),
        DataDriftPreset(),
        TextOverviewPreset(column_name="description") # Evaluates text length, sentiment drift, OOV
    ])
    
    report.run(reference_data=df_baseline, current_data=df_target, column_mapping=column_mapping)
    
    # 4. Save outputs for humans (HTML) and machines (JSON)
    report.save_html("advanced_eda_report.html")
    report.save_json("eda_metrics.json")
    
    # 5. Extract Programmatic Decision (Did the data drift?)
    with open("eda_metrics.json", "r") as f:
        metrics = json.load(f)
    
    # Traverse Evidently's JSON to find the dataset drift boolean
    # This structure depends on Evidently version, typically found in DataDriftPreset metrics
    drift_detected = False
    for metric in metrics["metrics"]:
        if metric["metric"] == "DatasetDriftMetric":
            drift_detected = metric["result"]["dataset_drift"]
            drift_share = metric["result"]["share_of_drifted_columns"]
            break
            
    print(f"Data Drift Detected: {drift_detected} (Share: {drift_share*100:.2f}%)")
    
    # 6. Log Data Health to MLflow (Creates a dashboard of your data quality over time)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Data_Profiling_Pipeline")
    
    with mlflow.start_run(run_name="Scheduled_EDA"):
        mlflow.log_metric("drift_share", drift_share)
        mlflow.log_metric("target_row_count", len(df_target))
        mlflow.log_metric("baseline_row_count", len(df_baseline))
        mlflow.log_param("drift_detected", str(drift_detected))
        mlflow.log_artifact("advanced_eda_report.html")
        
    return drift_detected, "advanced_eda_report.html"

def export_github_action_variable(is_drifted):
    """Writes the decision to GitHub Actions environment so the workflow knows what to do."""
    # GitHub uses GITHUB_OUTPUT to pass variables between workflow steps
    env_file = os.getenv('GITHUB_OUTPUT')
    if env_file:
        with open(env_file, "a") as f:
            f.write(f"trigger_retrain={str(is_drifted).lower()}\n")

if __name__ == "__main__":
    df_base, df_targ = fetch_data_splits()
    
    if df_base.empty or df_targ.empty:
        print("Insufficient data for drift comparison.")
        export_github_action_variable(False)
    else:
        is_drifted, report_path = run_advanced_profiling(df_base, df_targ)
        
        # Upload HTML to S3 as before (using boto3)
        if AWS_BUCKET:
            boto3.client('s3').upload_file(report_path, AWS_BUCKET, report_path)
            
        # Tell GitHub Actions whether to trigger the model training pipeline
        export_github_action_variable(is_drifted)
