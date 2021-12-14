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
    # Bash operators are used to execute bash commands
    # Dummy operator is used to check the status of the DAG and does nothing
    load_data = BashOperator(
        task_id="download_data",
        bash_command="python src/data/load_data.py"
    )

    # Task 2: Preprocessing the data
    # Python operators are used to execute python code
    split_data = BashOperator(
        task_id="preprocess_data",
        bash_command="python src/data/split_data.py"
    )

    # Task  : Send email notification
    # Email operators are used to send email notifications
    send_email = EmailOperator(
        task_id="send_email",
        to="ashish.tele@uconn.edu",
        subject="Churn Model Pipeline Notification",
        html_content="<h3> Dag Run Successfully </h3>"
    )

    load_data >> split_data >> send_email


