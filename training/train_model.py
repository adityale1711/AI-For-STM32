import os

from config import user_config
from common.utils.logs_utils import log_to_file
from object_detection.utils.models_mgt import ModelManagement


class TrainModel:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def train(self):
        saved_models_dir = os.path.join(self.output_dir, user_config.general.get('saved_models_dir'))
        tensorboard_log_dir = os.path.join(self.output_dir, user_config.general.get('logs_dir'))
        metrics_dir = os.path.join(self.output_dir, user_config.general.get('logs_dir'), 'metrics')

        # Log dataset and model info
        log_to_file(self.output_dir, f"Dataset: {user_config.dataset.get('name')}")
        if user_config.general.get('model_path'):
            log_to_file(self.output_dir, f"Model File: {user_config.general.get('model_path')}")
        elif user_config.training.get('resume_training_from'):
            log_to_file(self.output_dir, f"Resuming training from: {user_config.training.get('resume_training_from')}")

        model = ModelManagement().load_model_for_training()
