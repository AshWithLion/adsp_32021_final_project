import os
import sys
import logging
import traceback
import mlflow
import mlflow.pyfunc
import mlflow.h2o
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/train_model.log')
    ]
)
logger = logging.getLogger(__name__)


def run_train():
    try:
        # Detailed logging of environment and paths
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Python path: {sys.path}")

        # Initialize H2O with more robust initialization
        try:
            # Remove is_initialized check and use a try-except block
            logger.info("Attempting H2O initialization")
            h2o.init(max_mem_size="2G")
            logger.info("H2O initialized successfully")
        except Exception as h2o_init_error:
            logger.error(f"H2O initialization failed: {h2o_init_error}")
            raise

        # Robust file path handling
        base_path = os.path.abspath(os.path.dirname(__file__))
        project_root = os.path.dirname(base_path)
        data_path = os.path.join(project_root, 'data', 'processed_data.csv')

        logger.info(f"Attempting to read data from: {data_path}")

        # Verify file exists with detailed logging
        if not os.path.exists(data_path):
            logger.error(f"Data file not found at {data_path}")
            logger.error(f"Listing contents of {os.path.dirname(data_path)}:")
            logger.error(str(os.listdir(os.path.dirname(data_path))))
            raise FileNotFoundError(f"Data file not found at {data_path}")

        # Read and process data
        df = pd.read_csv(data_path)
        logger.info(f"Dataframe loaded. Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")

        target = 'price-transform'

        # Verify target column exists
        if target not in df.columns:
            logger.error(f"Target column '{target}' not found in dataframe")
            logger.error(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"Target column '{target}' not found")

        features = [col for col in df.columns if col != target]

        X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

        # Ensure data directory exists
        os.makedirs(os.path.join(project_root, 'data'), exist_ok=True)

        # Save split data
        X_train.to_csv(os.path.join(project_root, 'data', 'training_data.csv'), index=False)
        X_test.to_csv(os.path.join(project_root, 'data', 'test_data.csv'), index=False)
        y_test.to_csv(os.path.join(project_root, 'data', 'test_target.csv'), index=False, header=False)

        # Convert to H2O frames
        train_h2o = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
        test_h2o = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

        # AutoML training
        logger.info("Starting H2O AutoML training")
        aml = H2OAutoML(max_runtime_secs=300, seed=1)
        aml.train(y=target, training_frame=train_h2o)

        best_model = aml.leader

        # MLflow logging
        mlflow.set_experiment("automl_experiment")
        with mlflow.start_run(run_name="train_model"):
            mlflow.log_param("model_type", "H2O_AutoML")
            mlflow.h2o.log_model(best_model, "model")
            perf = best_model.model_performance(test_h2o)
            mlflow.log_metric("test_rmse", perf.rmse())
            mlflow.log_metric("test_mae", perf.mae())
            mlflow.log_metric("test_r2", perf.r2())

        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Error in training: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    run_train()