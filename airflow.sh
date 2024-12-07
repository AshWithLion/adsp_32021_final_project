# initialize the airflow
export AIRFLOW_HOME=~/airflow
airflow db init
airflow users create --username admin --password admin --firstname fname --lastname lname --role Admin --email admin@example.com

# Start Airflow Server
airflow webserver -p 8080
airflow scheduler