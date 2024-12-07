# scripts/train.py
import mlflow
import mlflow.pyfunc
import mlflow.h2o
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.model_selection import train_test_split

# Initialize H2O
h2o.init(max_mem_size="2G")

df = pd.read_csv('../data/processed_data.csv')
target = 'price-transform'
features = [col for col in df.columns if col != target]

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

X_train.to_csv('../data/training_data.csv', index=False)
# Save test sets for later inference
X_test.to_csv('../data/test_data.csv', index=False)
y_test.to_csv('../data/test_target.csv', index=False, header=False)

train_h2o = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
test_h2o = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

aml = H2OAutoML(max_runtime_secs=300, seed=1)
aml.train(y=target, training_frame=train_h2o)

best_model = aml.leader

mlflow.set_experiment("automl_experiment")
with mlflow.start_run(run_name="train_model"):
    mlflow.log_param("model_type", "H2O_AutoML")
    mlflow.h2o.log_model(best_model, "model")
    perf = best_model.model_performance(test_h2o)
    mlflow.log_metric("test_rmse", perf.rmse())
    mlflow.log_metric("test_mae", perf.mae())
    mlflow.log_metric("test_r2", perf.r2())