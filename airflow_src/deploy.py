import mlflow
import subprocess

def run_deploy(port=1234):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("automl_experiment")
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    latest_run_id = runs[0].info.run_id
    model_uri = f"runs:/{latest_run_id}/model"

    print("Starting MLflow model serving...")
    # Start MLflow serving in background
    subprocess.Popen(["mlflow", "models", "serve", "-m", model_uri, "-p", str(port)])
    print(f"Model is being served at http://127.0.0.1:{port}")