import argparse
import os
import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset

parser = argparse.ArgumentParser()
parser.add_argument("--data_version", default="original")
args = parser.parse_args()

target_column = "price-transform"  # Update to your actual target column

y_train = pd.read_csv('../data/training_target.csv', header=None).squeeze('columns')
y_test = pd.read_csv('../data/test_target.csv', header=None).squeeze('columns')
if args.data_version == 'changed':
    X_test = pd.read_csv('../data/test_data_changed.csv')
    predictions = pd.read_csv('../data/predictions_changed.csv', header=None).squeeze('columns')
else:
    X_test = pd.read_csv('../data/test_data.csv')
    predictions = pd.read_csv('../data/predictions_original.csv', header=None).squeeze('columns')

test_data = X_test.copy()
test_data['target'] = y_test
test_data['prediction'] = predictions

data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(current_data=test_data, reference_data=None)

reg_perf_report = Report(metrics=[RegressionPreset()])
reg_perf_report.run(current_data=test_data, reference_data=None)

# Save reports locally
os.makedirs('../reports', exist_ok=True)
drift_html = f"../reports/data_drift_report_{args.data_version}.html"
perf_html = f"../reports/regression_performance_report_{args.data_version}.html"

reference_data = pd.read_csv('../data/training_data.csv')
reference_data['target'] = y_train
reference_data['prediction'] = predictions

data_drift_report.run(current_data=test_data, reference_data=reference_data)
data_drift_report.save_html(drift_html)

reg_perf_report.run(current_data=test_data, reference_data=reference_data)
reg_perf_report.save_html(perf_html)

# Log to MLflow
mlflow.set_experiment("automl_experiment")
with mlflow.start_run(run_name=f"evidently_report_{args.data_version}", nested=True):
    mlflow.log_artifact(drift_html)
    mlflow.log_artifact(perf_html)

print(f"Evidently reports generated and logged to MLflow for data_version={args.data_version}.")