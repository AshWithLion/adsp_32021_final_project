from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from airflow_src.train import run_train
from airflow_src.deploy import run_deploy
from airflow_src.inference import run_inference
from airflow_src.evaluate import run_evaluate
from airflow_src.evidently_report import run_evidently_report
from airflow_src.change_data import run_change_data

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG('ml_pipeline',
         default_args=default_args,
         schedule_interval='@once',
         catchup=False) as dag:

    train_model = PythonOperator(
        task_id='train_model',
        python_callable=run_train)

    deploy_model = PythonOperator(
        task_id='deploy_model',
        python_callable=run_deploy,
        op_kwargs={'port': 1234} )

    inference_test = PythonOperator(
        task_id='inference_test',
        python_callable=run_inference,
        op_kwargs={'data_version': 'original'}
    )

    evaluate_results = PythonOperator(
        task_id='evaluate_results',
        python_callable=run_evaluate,
        op_kwargs={'data_version': 'original'}
    )

    monitoring_report_original = PythonOperator(
        task_id='monitoring_report_original',
        python_callable=run_evidently_report,
        op_kwargs={'data_version': 'original'}
    )

    change_data = PythonOperator(
        task_id='change_data',
        python_callable=run_change_data)

    inference_changed_test = PythonOperator(
        task_id='inference_changed_test',
        python_callable=run_inference,
        op_kwargs={'data_version': 'changed'}
    )

    evaluate_changed_results = PythonOperator(
        task_id='evaluate_changed_results',
        python_callable=run_evaluate,
        op_kwargs={'data_version': 'changed'}
    )

    monitoring_report_changed = PythonOperator(
        task_id='monitoring_report_changed',
        python_callable=run_evidently_report,
        op_kwargs={'data_version': 'changed'}
    )

    # Define workflow
    (
        train_model
        >> deploy_model
        >> inference_test
        >> evaluate_results
        >> monitoring_report_original
        >> change_data
        >> inference_changed_test
        >> evaluate_changed_results
        >> monitoring_report_changed
    )