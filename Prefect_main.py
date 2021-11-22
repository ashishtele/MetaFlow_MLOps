from prefect import Flow, task
from typing import List, Dict, Any
import prefect
import pandas as pd
from sklearn.model_selection import train_test_split

@task
def load_data(external_path: str, 
                raw_data_path: str, 
                model_var: List[str]):
    data = pd.read_csv(external_path,
                       sep=',',
                       encoding='utf-8')
    data = data[model_var]
    data.to_csv(raw_data_path, index=False)
    return None

@task
def split_data(raw_data_path: str,
                train_data_path: str, 
                test_data_path: str, 
                split_ratio: float, 
                random_state):
    data = pd.read_csv(raw_data_path,
                       sep=',',
                       encoding='utf-8')
    data_train, data_test = train_test_split(data,
                                                test_size=split_ratio,
                                                random_state=random_state)
    data_train.to_csv(train_data_path, index=False)
    data_test.to_csv(test_data_path, index=False)
    return None
    

with Flow("Prefect_MLOps") as flow:

    external_path = 'data/external/train.csv'
    raw_data_path = 'data/raw/train.csv'
    model_var = ['churn','number_vmail_messages','total_day_calls','total_eve_minutes','total_eve_charge','total_intl_minutes','number_customer_service_calls']
    
    train_data_path = 'data/processed/train.csv'
    test_data_path = 'data/processed/test.csv'
    split_ratio = 0.3
    random_state = 42

    load_data(external_path, raw_data_path, model_var)
    split_data(raw_data_path,train_data_path, test_data_path, split_ratio, random_state)

#flow.run()
flow.register(project_name="Prefect_MLOp")