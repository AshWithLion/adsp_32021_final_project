# scripts/inference.py
import argparse
import requests
import pandas as pd
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data_version", default="original", help="Use 'original' or 'changed' dataset.")
args = parser.parse_args()

if args.data_version == 'changed':
    data_path = '../data/test_data_changed.csv'
    pred_path = '../data/predictions_changed.csv'
else:
    data_path = '../data/test_data.csv'
    pred_path = '../data/predictions_original.csv'

X_test = pd.read_csv(data_path)

url = "http://127.0.0.1:1234/invocations"
headers = {"Content-Type": "application/json"}
data_json = X_test.to_dict(orient='records')
payload = {
    "dataframe_records": data_json
}
response = requests.post(url, headers=headers, data=json.dumps(payload))
predictions = response.json()

predictions = predictions["predictions"]  # This should be a list of floats
float_predictions = [p["predict"] for p in predictions]

# Now predictions should be a list of numeric values, one per test sample
pd.Series(float_predictions).to_csv(pred_path, index=False, header=False)

print(f"Inference completed for {args.data_version}, predictions saved to {pred_path}.")