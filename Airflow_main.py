## Importing the libraries
# Instantiate the DAG
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta
from airflow.operators.email_operator import EmailOperator
from airflow.operators.python_operator import PythonOperator


default_args = {
    'owner': 'MLOps_admin',
    'depends_on_past': False,
    "email_on_failure": False,
    "email_on_retry": False,
    "email": ['ashish.tele@uconn.edu'],
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

with DAG("churn_model_pipeline",
        description="Churn Model Pipeline",
        default_args=default_args,
        schedule_interval= "@daily",
        start_date=datetime(2021, 12, 12),
        catchup = False
        ) as dag:

    # Task 1: Downloading the data
    download_data = BashOperator(
        task_id="download_data",
        bash_command="python src/download_data.py"
    )

    # Task 2: Preprocessing the data
    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command="python src/preprocess_data.py"
    )

    # Task  : Send email notification
    send_email = EmailOperator(
        task_id="send_email",
        to="ashish.tele@uconn.edu",
        subject="Churn Model Pipeline Notification",
        html_content="<h3> Dag Run Successfully </h3>"
    )


