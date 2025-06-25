from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='loan default prediction pipeline (monthly)',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1), # full 2 years 
    catchup=True,
    max_active_runs=1
) as dag:

    # data pipeline
    data_processing_start = DummyOperator(task_id="data_processing_main_start")

    # Generate bronze, silver, gold data tables based on raw data received for that month
    data_processing_run = BashOperator(
        task_id='data_processing_main_run',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 data_processing_main.py ' # map to correct python 3 explicitly to avoid assumptions
            '--snapshotdate "{{ ds }}"'
        ),
    )

    data_processing_complete = DummyOperator(task_id="data_processing_main_complete")

    # --- model inference ---
    model_inference_start = DummyOperator(task_id="model_inference_start")

    model_inference_run = BashOperator(
        task_id='model_inference_run',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_inference.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname "credit_model_2024_09_01.pkl"' # hardcode chosen model for now, will manually change if train better model in future
        ),
    )

    model_inference_complete = DummyOperator(task_id="model_inference_complete")
    
    # Define task dependencies to run scripts sequentially
    data_processing_start >> data_processing_run >> data_processing_complete >> model_inference_start >> model_inference_run >> model_inference_complete


    # --- model monitoring ---
    model_monitor_start = DummyOperator(task_id="model_monitor_start")

    model_monitor_run = BashOperator(
        task_id='model_monitor_run',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_monitoring.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname "credit_model_2024_09_01.pkl"' # hardcode chosen model for now, will manually change if train better model in future
        ),
    )

    model_monitor_complete = DummyOperator(task_id="model_monitor_complete")
    
    # Define task dependencies to run scripts sequentially
    model_inference_complete >> model_monitor_start >> model_monitor_run >> model_monitor_complete