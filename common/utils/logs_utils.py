import os
import mlflow
import datetime


def mlflow_init(project_name, uri, operation_mode):
    """
    Initializes an MLFlow experiment with the specified tracking URI and experiment name.

    Args:
        project_name (str): Name of the MLFlow experiment.
        uri (str): MLFlow tracking server URI.
        operation_mode (str): Mode of operation, stored as a parameter.

    Returns:
        None
    """

    # Store operation mode as a parameter for logging
    params = {'operation_mode': operation_mode}

    # Set the MLFlow tracking server URI
    mlflow.set_tracking_uri(uri)

    # Set or create an MLFlow experiment
    mlflow.set_experiment(project_name)

    # Assign a unique run name based on the current timestamp
    mlflow.set_tag('mlflow.runName', 'Trial-' + str(datetime.datetime.now()))

    # Log the operation mode as an MLFlow parameter
    mlflow.log_params(params)

    # Enable automatic logging for TensorFlow, but disable model logging
    mlflow.tensorflow.autolog(log_models=False)


def log_to_file(dir: str, log: str) -> None:
    """
    Appends a log message to a file named 'main.log' in the specified directory.

    Args:
        dir (str): The directory where the log file is located.
        log (str): The log message to be written.

    Returns:
        None
    """

    # Open the log file in append mode ('a') to avoid overwriting existing logs
    with open(os.path.join(dir, 'main.log'), 'a') as log_file:
        # Write the log message with a newline for proper formatting
        log_file.write(log + '\n')
