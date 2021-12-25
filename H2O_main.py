import h2o
from h2o.automl import H2OAutoML, get_leaderboard
import argparse
import yaml
import json
import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="H2O AutoML")

    parser.add_argument("--config", 
                        default="params.yaml")
    return parser.parse_args()

def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def main():
    args = parse_args()

    h2o.init()
    client = MlflowClient()

    config = read_params(args.config)
    mlflow_config = config["mlflow_config"]

    # New experiment name for H2o experiment
    exp_name = mlflow_config["experiment_name_h2o"]

    # Crate experiment for MLFLOW
    try:
        experiment_id = mlflow.create_experiment(exp_name)
        experiment = client.get_experiment_by_name(exp_name)
    except:
        experiment = client.get_experiment_by_name(exp_name)
    
    mlflow.set_experiment(exp_name)

    print('Printing experiment Details ------------------------------------------------')
    print(f'Experiment Name: {exp_name}')
    print(f'Experiment ID: {experiment.experiment_id}')
    print(f'Artifact Location: {experiment.artifact_location}')
    print(f'Tracking URI: {mlflow.get_tracking_uri()}')

    # Importing data in H2O dataframe
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]

    target = config["raw_data_config"]["target"]
    predictors = config["raw_data_config"]["model_var"]

    train = h2o.import_file(train_data_path)
    test = h2o.import_file(test_data_path)

    # Determine the shapes of train and test data
    print('Printing train and test data shapes ------------------------------------------------')
    print(f'Train data shape: {train.shape}')
    print(f'Test data shape: {test.shape}')

    # Column types dump for matching with test dataframe
    with open('data/processed/train_column_types.json', 'w') as f:
        json.dump(train.types, f)

    # Change target variable to factor
    train[target] = train[target].asfactor()

    ########################### MLFLOW ###########################
    with mlflow.start_run(run_name = mlflow_config['run_name_h2o']) as mlops_run:
        auto_ml = H2OAutoML(
            seed = mlflow_config['seed'],
            max_models = mlflow_config['max_models'],
            balance_classes = mlflow_config['balance_classes'],
            sort_metric = mlflow_config['sort_metric'],
            verbosity = mlflow_config['verbosity'],
            exclude_algos = mlflow_config['exclude_algos']
        )

        # Train the model
        auto_ml.train(
            x = predictors,
            y = target,
            training_frame = train)

        # Log the metrics
        mlflow.log_metric('log_loss', auto_ml.leader.logloss())
        mlflow.log_metric('auc', auto_ml.leader.auc())

        # Log best model
        mlflow.h2o.log_model(auto_ml.leader, "best_model")
        model_uri = mlflow.get_artifact_uri("best_model")

        print(f'Best model URI: {model_uri}')

        # Experiment run IDs
        exp_id = experiment.experiment_id
        run_id = mlflow.active_run().info.run_id

        # Save leaderboard to a CSV files
        leaderboard = get_leaderboard(auto_ml, extra_columns = 'ALL')
        leaderboard_path = f'mlruns/{exp_id}/{run_id}/leaderboard.csv'
        leaderboard.as_data_frame().to_csv(leaderboard_path, index = False)
        print(f'Leaderboard saved to {leaderboard_path}')

if __name__ == '__main__':
    main()
