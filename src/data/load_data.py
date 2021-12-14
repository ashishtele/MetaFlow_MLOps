# Importing the libraries
import numpy as np
import pandas as pd
import yaml
import argparse
import typing

# create a function to read params from yaml files
def read_params(config_path: str):
    """
    Reads the parameters from the .yaml file
    input: params.yaml location
    output: paramaeters in a dictionary
    """

    with open(config_path, 'r') as yaml_file:
        try:
            config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)
    return config

# create a function to read data from csv files
def load_data(data_path: str, data_type = 'train', model_var: typing.List[str] = None):
    """
    Reads the data from the .csv file from given path
    input: data_path, data_type
    output: data in a pandas dataframe
    """
    if data_type == 'train':
        data = pd.read_csv(data_path, 
                            sep=',', 
                            encoding='utf-8')
        data = data[model_var]
        return data


# create a function to load raw data from path
def load_raw_data(config_path: str):
    """
    Load datafrom external location (data/external) tp raw folder (data/raw)
    with train and test data
    input: data_path
    output: save file in data/raw folder
    """
    config = read_params(config_path)
    external_data_path = config['external_data_config']['external_data_csv']
    raw_data_path = config['raw_data_config']['raw_data_csv']
    model_var = config['raw_data_config']['model_var']

    df = load_data(data_path=external_data_path, 
                    data_type='train', 
                    model_var=model_var)
    df.to_csv(raw_data_path, index = False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                        type=str, 
                        default='params.yaml')
    args = parser.parse_args()
    config_path = args.config
    load_raw_data(config_path)