import os
import mlflow
import datetime


def mlflow_init(project_name, uri, operation_mode):
    params = {'operation_mode': operation_mode}

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(project_name)
    mlflow.set_tag('mlflow.runName', 'Trial-' + str(datetime.datetime.now()))
    mlflow.log_params(params)
    mlflow.tensorflow.autolog(log_models=False)


def log_to_file(dir: str, log: str) -> None:
    with open(os.path.join(dir, 'main.log'), 'a') as log_file:
        log_file.write(log + '\n')
