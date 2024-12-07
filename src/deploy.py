import mlflow
import subprocess

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("automl_experiment")
runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
latest_run_id = runs[0].info.run_id
model_uri = f"runs:/{latest_run_id}/model"

print("Serving the model...")
# This will run a blocking process. In production, do this in a separate terminal
subprocess.Popen(["mlflow", "models", "serve", "-m", model_uri, "-p", "1234"])