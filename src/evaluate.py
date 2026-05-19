import os
import mlflow
from mlflow.tracking import MlflowClient

# Environment Configuration
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

def get_latest_run(experiment_name="Real_Estate_Intelligence_Ensemble"):
    """Fetches the metrics of the model we just trained (The Challenger)."""
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found.")
        
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    if not runs:
        raise ValueError("No recent runs found to evaluate.")
        
    return runs[0]

def get_production_run(model_name="regression_ensemble"):
    """Fetches the metrics of the currently deployed model (The Champion)."""
    try:
        # Get the version currently tagged as "Production"
        latest_versions = client.get_latest_versions(name=model_name, stages=["Production"])
        if not latest_versions:
            return None
            
        prod_version = latest_versions[0]
        prod_run_id = prod_version.run_id
        return client.get_run(prod_run_id)
    except Exception as e:
        print(f"No existing production model found or error accessing registry: {e}")
        return None

def evaluate_and_promote():
    """Compares Challenger vs Champion and promotes if better."""
    print("Initiating Champion vs Challenger Evaluation...")
    
    challenger_run = get_latest_run()
    challenger_r2 = challenger_run.data.metrics.get("reg_r2", 0)
    challenger_acc = challenger_run.data.metrics.get("clf_accuracy", 0)
    
    print(f"Challenger Metrics -> R2: {challenger_r2:.4f}, Accuracy: {challenger_acc:.4f}")
    
    champion_run = get_production_run()
    
    promote = False
    
    if not champion_run:
        print("No Champion found in production. Automatically promoting Challenger.")
        promote = True
    else:
        champion_r2 = champion_run.data.metrics.get("reg_r2", 0)
        champion_acc = champion_run.data.metrics.get("clf_accuracy", 0)
        print(f"Champion Metrics   -> R2: {champion_r2:.4f}, Accuracy: {champion_acc:.4f}")
        
        # Promotion Logic: We require improvement in the primary regression metric
        # without catastrophically degrading the classification metric.
        r2_improvement = challenger_r2 > champion_r2
        acc_stable = challenger_acc >= (champion_acc - 0.02) # Allow 2% tolerance
        
        if r2_improvement and acc_stable:
            print("Challenger defeated Champion! Flagging for promotion.")
            promote = True
        else:
            print("Challenger failed to defeat Champion. Deployment aborted.")
            promote = False

    # Export decision to GitHub Actions environment
    env_file = os.getenv('GITHUB_OUTPUT')
    if env_file:
        with open(env_file, "a") as f:
            f.write(f"deploy_approved={str(promote).lower()}\n")
            f.write(f"run_id={challenger_run.info.run_id}\n")

if __name__ == "__main__":
    evaluate_and_promote()
