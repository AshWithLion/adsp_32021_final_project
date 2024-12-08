import argparse
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_evaluate(data_version="original"):
    y_test = pd.read_csv('../data/test_target.csv', header=None).squeeze('columns')
    if data_version == 'changed':
        y_pred = pd.read_csv('../data/predictions_changed.csv', header=None).squeeze('columns')
    else:
        y_pred = pd.read_csv('../data/predictions_original.csv', header=None).squeeze('columns')

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.set_experiment("automl_experiment")
    with mlflow.start_run(run_name=f"evaluate_{data_version}", nested=True):
        mlflow.log_param("evaluation_data_version", data_version)
        mlflow.log_metric("eval_rmse", rmse)
        mlflow.log_metric("eval_mae", mae)
        mlflow.log_metric("eval_r2", r2)

    print(f"Evaluation for {data_version}: RMSE={rmse}, MAE={mae}, R2={r2}")