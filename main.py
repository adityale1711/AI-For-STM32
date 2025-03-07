import os
import mlflow
import argparse
import tensorflow as tf

from config import user_config
from training.train_model import TrainModel
from common.utils.cfg_utils import get_random_seed
from common.utils.gpu_utils import set_gpu_memory_limit
from common.utils.logs_utils import mlflow_init, log_to_file


class STM32AI:
    def __init__(self, output_dir, operation_mode):
        self.output_dir = output_dir
        self.operation_mode = user_config.operation_modes[operation_mode]

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description='AI for STM32')

        parser.add_argument(
            '--output_dir', type=str, default='outputs', help='Output Directory'
        )
        parser.add_argument(
            '--operation_mode', type=int, default=0, help='Operation Mode'
        )

        args = parser.parse_args()

        return args

    def run(self):
        if user_config.gpu_memory_limit is not None:
            set_gpu_memory_limit(user_config.gpu_memory_limit)
        else:
            print(
                '[WARNING] The usable GPU memory is unlimited\n'
                'Please consider setting the "gpu_memory_limit" attribute in the "general" section of your'
                'configuration file.'
            )

        # Create if output directory is not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Initialize MLFlow Tracking
        mlflow_init(user_config.project_name, user_config.mlflow_uri, self.operation_mode)

        # Random seed generator
        seed = get_random_seed(user_config.global_seed)

        print(f'[INFO] The random seed for this simulation is {seed}')
        if seed is not None:
            tf.keras.utils.set_random_seed(seed)

        mode = self.operation_mode

        mlflow.log_param('model_path', user_config.model_path)
        log_to_file(self.output_dir, f'operation_mode: {mode}')

        if mode == 'training':
            TrainModel(self.output_dir).train()
            print('[INFO] Training Complete')


def main():
    args = STM32AI.parse_arguments()
    app = STM32AI(args.output_dir, args.operation_mode)
    app.run()


if __name__ == '__main__':
    main()
