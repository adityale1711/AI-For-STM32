import os

from config import user_config
from common.utils.logs_utils import log_to_file


class TrainModel:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def train(self):
        saved_models_dir = os.path.join(self.output_dir, user_config.saved_models_dir)
        tensorboard_log_dir = os.path.join(self.output_dir, user_config.logs_dir)
        metrics_dir = os.path.join(self.output_dir, user_config.logs_dir, 'metrics')

        # Log dataset and model info
        log_to_file(self.output_dir, f'Dataset: {user_config.dataset_name}')
        if user_config.model_path is not None:
            log_to_file(self.output_dir, f'Model File: {user_config.model_path}')
        elif user_config.resume_training_from is not None:
            log_to_file(self.output_dir, f'Resuming training from: {user_config.resume_training_from}')
