from config import user_config
from common.utils.cfg_utils import check_attributes
from common.utils.models_utils import check_model_support


class ModelManagement:
    @staticmethod
    def check_st_yolo_x(cft, model_type, random_resizing=None):
        check_attributes(cft, expected=['input_shape'], optional=['depth_mul', 'width_mul'], section='training.model')

        if random_resizing:
            raise ValueError(f'\nrandom_periodic_resizing is not supported for model "{model_type}".\n'
                             f'Please check the "data_augmentation" section of your configuration file.')

    def get_zoo_model(self):
        supported_models = {
            'st_yolo_x': None
        }

        model_name = user_config.general.get('model_type')
        message = '\nPlease check the "general" section of your configuration file.'
        check_model_support(model_name, supported_models=supported_models, message=message)

        cft = user_config.training.get('model')
        input_shape = cft.get('input_shape')
        num_classes = len(user_config.dataset.get('class_name'))
        random_resizing = True if user_config.data_augmentation and user_config.data_augmentation.get(
            'random_periodic_resizing'
        ) else False
        section = 'training.model'
        model = None

        if model_name == 'st_yolo_x':
            self.check_st_yolo_x(cft, 'st_yolo_x', random_resizing=random_resizing)
            num_anchors = len(user_config.postprocessing.get('yolo_anchors'))

            if not cft.get('depth_mull') and cft.get('width_mul'):
                cft['depth_mul'] = 0.33
                cft['width_mul'] = 0.25

    def load_model_for_training(self):
        model_type = user_config.general.get('model_type')
        model = None

        # Train a model from the model zoo
        if user_config.training.get('model'):
            print(f'[INFO] Loading model from Model Zoo: {model_type}')
            model = self.get_zoo_model()
